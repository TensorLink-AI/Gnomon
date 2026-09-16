"""Host-only authenticated TSFM client; requests and receipts never contain keys."""
import hashlib,json,math,re,time
from pathlib import Path
import urllib.request,urllib.error

BASE_URL='https://chris-t-ensor-paracast.chutes.ai'
PROVIDER='paracast_route'
PROVIDERS=('paracast_route','paracast_ensemble2')
POLICIES={p:{'mode':'route' if p=='paracast_route' else 'ensemble',**({} if p=='paracast_route' else {'top_k':2}),
    'freq':'D','horizon':14,'quantiles':[0.1,0.5,0.9],'return_members':True} for p in PROVIDERS}

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def parse_forecast(response,horizon):
    if not isinstance(response,dict):raise ValueError('Expected JSON object')
    forecasts=response.get('forecasts')
    if not isinstance(forecasts,list) or len(forecasts)!=1:raise ValueError('Expected one forecast')
    if not isinstance(forecasts[0],dict) or not isinstance(forecasts[0].get('quantiles'),dict):raise ValueError('Missing quantiles')
    point=forecasts[0]['quantiles'].get('0.5')
    if not isinstance(point,list) or len(point)!=horizon or any(type(v) not in (float,int) or not math.isfinite(v) for v in point):
        raise ValueError('Invalid univariate median or horizon')
    meta=response.get('meta')
    if not isinstance(meta,dict) or not meta.get('models_used'):raise ValueError('Missing model provenance')
    return {'point':[max(0.,float(v)) for v in point],'clipped_points':sum(v<0 for v in point),
            'api_meta':meta,'point_semantics':'nonnegative-clipped API median (0.5 quantile)'}

class Client:
    def __init__(self,key_path,output):
        match=re.search(r'cpk_[A-Za-z0-9_.-]+',Path(key_path).read_text())
        if not match:raise ValueError('Chutes credential not found')
        self._key=match.group(0);self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
    def call(self,endpoint,payload=None,label=None):
        if endpoint not in ('health','models','forecast'):raise ValueError('Endpoint not authorized by this experiment')
        path=self.output/(label or endpoint)
        if path.exists():raise ValueError('Refusing to overwrite an API attempt')
        path.mkdir()
        if payload is not None:(path/'request.json').write_text(json.dumps(payload,allow_nan=False))
        req=urllib.request.Request(BASE_URL+'/'+endpoint,data=None if payload is None else json.dumps(payload,allow_nan=False).encode(),
            headers={'Authorization':'Bearer '+self._key,'Content-Type':'application/json'})
        before=time.monotonic();started=time.time();raw=b'';status=None;category=None
        try:
            with urllib.request.urlopen(req,timeout=75) as response:status=response.status;raw=response.read(16*1024*1024)
        except urllib.error.HTTPError as exc:status=exc.code;raw=exc.read(1000000);category='http_error'
        except Exception as exc:category=type(exc).__name__
        raw=raw.replace(self._key.encode(),b'[REDACTED]')
        (path/'response.json').write_bytes(raw)
        receipt={'endpoint':endpoint,'started_at_epoch':started,'seconds':time.monotonic()-before,
            'http_status':status,'error_category':category,'response_sha256':hashlib.sha256(raw).hexdigest(),
            'request_sha256':digest(payload) if payload is not None else None,'attempts':1,'automatic_retries':0}
        (path/'receipt.json').write_text(json.dumps(receipt,indent=2))
        if status!=200:raise ValueError('TSFM request failed: '+str(status or category))
        return json.loads(raw)
    def forecast(self,history,label,provider):
        return self.call('forecast',{**POLICIES[provider],'series':[list(map(float,history))]},label)
    def catalog(self,label='models'):
        """Advisory catalog: retain failure receipts without discarding forecasts."""
        try:return {'available':True,'catalog':self.call('models',label=label)}
        except ValueError:return {'available':False,'receipt':label+'/receipt.json','automatic_retries':0}
