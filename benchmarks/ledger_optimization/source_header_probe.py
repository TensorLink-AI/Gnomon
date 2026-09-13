"""Bounded acquisition of pinned source archives; inspect TSF headers only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

SOURCES = {
    'electricity': ('4656140', 'electricity_hourly_dataset.zip', '18096614662b02640d265ad2a6a416bd'),
    'pedestrian': ('4656626', 'pedestrian_counts_dataset.zip', '420043b57ed8577564d299742e8acf97'),
}


def header(stream, encoding='cp1252'):
    lines=[]
    for _ in range(200):
        line=stream.readline(8193)
        if not line or len(line)>8192:
            raise ValueError('Missing or oversized TSF header')
        text=line.decode(encoding,errors='strict').rstrip('\r\n')
        lines.append(text)
        if text.strip().lower()=='@data':
            return lines
    raise ValueError('TSF header exceeds bounded scan')


def probe(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=[]
    for name,(record,filename,expected) in SOURCES.items():
        url=f'https://zenodo.org/records/{record}/files/{filename}?download=1'
        receipt={'source':name,'url':url,'expected_md5':expected,'observation_rows_parsed':0,
                 'header_encoding':'cp1252'}
        started=time.monotonic()
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'Gnomon-development-source-probe/1'})
            path=output/filename
            with urllib.request.urlopen(request,timeout=30) as response, path.open('xb') as target:
                receipt['http_status']=response.status
                total=0
                while chunk:=response.read(65536):
                    total+=len(chunk)
                    if total>64*1024*1024:raise ValueError('Archive exceeds 64 MiB bound')
                    target.write(chunk)
            data=path.read_bytes()
            receipt.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),md5=hashlib.md5(data).hexdigest())
            if receipt['md5']!=expected:raise ValueError('Publisher checksum mismatch')
            with zipfile.ZipFile(path) as archive:
                members=[m for m in archive.infolist() if m.filename.endswith('.tsf')]
                if len(members)!=1:raise ValueError('Expected exactly one TSF member')
                with archive.open(members[0]) as stream:receipt['header']=header(stream)
                receipt['member']=members[0].filename
            receipt['status']='header_verified'
        except Exception as error:
            receipt.update(status='failed',error=type(error).__name__,message=str(error))
        receipt['seconds']=time.monotonic()-started
        records.append(receipt)
        (output/'receipt.json').write_text(json.dumps({'sources':records,'api_model_calls':0},indent=2)+'\n')
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output');args=parser.parse_args()
    print(json.dumps(probe(args.output),indent=2))
