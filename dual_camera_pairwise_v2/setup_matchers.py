"""Reproduce pinned inference-only matcher dependencies inside this extension.

Run with the repository's unchanged .venv/bin/python -B. Network access is
required only when a pinned artifact is absent. No original package is installed,
upgraded, removed, or patched. Existing matching artifacts are hash checked.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile

from pairwise.runtime import configure_runtime,EXT_ROOT,sha256,write_json,new_directory
configure_runtime()
from pairwise.matching import SOURCES

PACKAGES=['einops==0.8.1','rich==14.2.0','markdown-it-py==4.0.0','mdurl==0.1.2','pygments==2.19.2']
DINO_COMMIT='adc254450203739c8149213a7a69d8d905b4fcfa'


def download(url,path,attempts):
    part=path.with_suffix(path.suffix+'.part')
    try:
        with urllib.request.urlopen(url,timeout=60) as response, part.open('xb') as target:
            while block:=response.read(8*1024*1024):target.write(block)
        part.rename(path);attempts.append({'url':url,'status':'downloaded','sha256':sha256(path)})
        return True
    except (OSError,urllib.error.URLError) as error:
        if part.exists():part.unlink()
        attempts.append({'url':url,'status':'failed','reason':repr(error)})
        return False


def main():
    report=new_directory('reports','matcher_setup');attempts=[]
    deps=EXT_ROOT/'environments/deps';deps.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(deps))
    missing=[]
    for package in PACKAGES:
        name,version=package.split('==')
        try:current=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:current=None
        if current!=version:missing.append(package)
    if missing:
        with (report/'pip.log').open('x') as log:
            subprocess.run([sys.executable,'-B','-m','pip','install','--target',str(deps),
                            '--no-compile','--no-deps',*missing],env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
    for backend,name in [('roma_v2','RoMaV2')]:
        source=SOURCES[backend];path=EXT_ROOT/'vendor'/name
        if not path.exists():
            subprocess.run(['git','clone',source['repo']+'.git',str(path)],check=True)
            subprocess.run(['git','-C',str(path),'checkout',source['commit']],check=True)
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if actual!=source['commit']:raise ValueError(f'Existing source commit differs: {path}')
        weight=EXT_ROOT/'weights'/source['checkpoint'];weight.parent.mkdir(parents=True,exist_ok=True)
        if not weight.exists():
            ok=False
            if not ok:ok=download(source['checkpoint_url'],weight,attempts)
            if not ok:raise RuntimeError(f'blocked_missing_pretrained_weights: {backend}')
        if sha256(weight)!=source['checkpoint_sha256_expected']:raise ValueError(f'Pinned checkpoint hash differs: {weight}')
    archive=EXT_ROOT/'vendor'/f'dinov3_{DINO_COMMIT}.zip'
    folder=EXT_ROOT/'vendor'/f'dinov3-{DINO_COMMIT}'
    if not folder.exists():
        if not archive.exists() and not download(f'https://github.com/facebookresearch/dinov3/archive/{DINO_COMMIT}.zip',archive,attempts):
            raise RuntimeError('blocked_matcher_dependency: DINOv3 source unavailable')
        with zipfile.ZipFile(archive) as zipped:zipped.extractall(EXT_ROOT/'vendor')
    hub=EXT_ROOT/'cache/torch/hub';hub.mkdir(parents=True,exist_ok=True)
    links={hub/f'facebookresearch_dinov3_{DINO_COMMIT}':folder,
           hub/'checkpoints/romav2.0.1.pt':EXT_ROOT/'weights/romav2.0.1.pt'}
    for link,target in links.items():
        link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():link.symlink_to(target,target_is_directory=target.is_dir())
        elif link.resolve()!=target.resolve():raise ValueError(f'Unexpected existing torch cache target: {link}')
    result={'status':'ready','source':{'roma_v2': SOURCES['roma_v2']},'dependency_target':str(deps),'package_pins':PACKAGES,
            'network_attempts':attempts,'original_environment_modified':False,
            'source_backbone_commit':DINO_COMMIT,'backbone_archive_sha256':sha256(archive)}
    write_json(report/'setup.json',result)
    print(json.dumps({'status':'ready','report':str(report/'setup.json')}))


if __name__=='__main__':main()
