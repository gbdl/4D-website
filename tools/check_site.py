#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Check portable HTML links, identifiers and required static assets."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import argparse
import json
import hashlib
from update_audit_display import check_or_update
class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.ids=[]
    def handle_starttag(self, tag, attrs):
        d=dict(attrs)
        if 'id' in d: self.ids.append(d['id'])
        for name in ('href','src'):
            if name in d: self.links.append(d[name])
def main():
    p=argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]); p.add_argument('--output', type=Path)
    args=p.parse_args(); root=args.root.resolve(); failed=[]; count=0
    html=[root/'index.html', *sorted((root/'data/tables').glob('*.html'))]
    for path in html:
        parser=Links(); parser.feed(path.read_text()); count+=len(parser.links)
        if len(parser.ids)!=len(set(parser.ids)): failed.append(f'Duplicate ids: {path.relative_to(root)}')
        for link in parser.links:
            u=urlsplit(link)
            if u.scheme or u.netloc: continue
            if u.path.startswith('/'): failed.append('Absolute site path: '+link); continue
            target=(path.parent/unquote(u.path)).resolve() if u.path else path
            if not target.is_relative_to(root) or not target.exists(): failed.append(f'Missing/escaping link: {link}')
            elif target==path and u.fragment and u.fragment not in parser.ids: failed.append(f'Missing anchor: {link}')
    for name in ('metrics','experiments','audit','audit-supplement'):
        json.loads((root/f'data/{name}.json').read_text())
    try:
        check_or_update(root,check=True)
    except (AssertionError,ValueError,KeyError) as e:
        failed.append(str(e))
    report={'status':'failed' if failed else 'passed','html_files':len(html),'links_checked':count,'failures':failed,'index_sha256':hashlib.sha256((root/'index.html').read_bytes()).hexdigest(),'scope':'Local links and JSON parse checks; visual and scientific reviews are separate.'}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(bool(failed))
if __name__=='__main__': main()
