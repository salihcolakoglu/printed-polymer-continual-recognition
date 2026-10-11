# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Salih Colakoglu and Mustafa Onat
"""Trim MATLAB PDF page margins and preserve heatmap SVG as vector cells.

No scientific values or PNG outputs change. The MATLAB originals are retained.
"""
from pathlib import Path
import base64
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

folder=Path(__file__).resolve().parent
backup=folder/"_vector_export_originals_v1"
backup.mkdir(exist_ok=False)
report={"PDF_crop":[],"SVG_vector_replacements":[],"new_training":0,"MAT_read":0}
for path in sorted(folder.glob("Figure_*.pdf")):
    reader=PdfReader(path)
    assert len(reader.pages)==1
    page=reader.pages[0]
    assert len(page.images)==0
    contents=page.get_contents().get_data().decode("latin1")
    assert "0.750000000 0 0 -0.750000000 0 842 cm" in contents[:150]
    match=re.search(r"1 1 1 scn\n/GSa gs\n([\d.]+) ([\d.]+) m\n([\d.]+) ([\d.]+) l\n([\d.]+) ([\d.]+) l",contents)
    assert match
    x0,y0,x1,_,_,y1=map(float,match.groups())
    box=[.75*x0,842-.75*y1,.75*x1,842-.75*y0]
    assert box[2]>box[0] and box[3]>box[1]
    original_bytes=path.read_bytes()
    page.mediabox=RectangleObject(box)
    page.cropbox=RectangleObject(box)
    writer=PdfWriter();writer.add_page(page)
    path.rename(backup/path.name)
    with path.open("xb") as stream:writer.write(stream)
    report["PDF_crop"].append({"file":path.name,"original_sha256":hashlib.sha256(original_bytes).hexdigest(),
                               "final_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"crop_box":box,
                               "raster_images":0})

ns="http://www.w3.org/2000/svg"
xlink="http://www.w3.org/1999/xlink"
ET.register_namespace("",ns);ET.register_namespace("xlink",xlink)
path=folder/"Figure_4_paired_old_task_gain.svg"
tree=ET.parse(path);root=tree.getroot();parents={child:parent for parent in root.iter() for child in parent}
images=[element for element in root.iter() if element.tag=="{"+ns+"}image"]
assert len(images)==2
for index,element in enumerate(images):
    href=element.attrib["{"+xlink+"}href"]
    with Image.open(io.BytesIO(base64.b64decode(href.split(",",1)[1]))) as image:
        image=image.convert("RGB")
        width=float(element.attrib["width"]);height=float(element.attrib["height"])
        x=float(element.attrib.get("x",0));y=float(element.attrib.get("y",0))
        group=ET.Element("{"+ns+"}g",{"data-vector-replacement":"fixed heatmap cells" if index==0 else "colour scale"})
        nrows,ncols=(4,5) if index==0 else (256,1)
        for row in range(nrows):
            for col in range(ncols):
                px=min(image.width-1,int((col+.5)*image.width/ncols))
                py=min(image.height-1,int((row+.5)*image.height/nrows))
                color=image.getpixel((px,py))
                ET.SubElement(group,"{"+ns+"}rect",{
                    "x":f"{x+col*width/ncols:.8f}","y":f"{y+row*height/nrows:.8f}",
                    "width":f"{width/ncols:.8f}","height":f"{height/nrows:.8f}",
                    "fill":"#%02x%02x%02x"%color,"stroke":"none"})
    parent=parents[element];position=list(parent).index(element);parent.remove(element);parent.insert(position,group)
    report["SVG_vector_replacements"].append({"original_element":index,"vector_rectangles":nrows*ncols})
path.rename(backup/path.name)
tree.write(path,encoding="utf-8",xml_declaration=True)
assert not [element for element in root.iter() if element.tag=="{"+ns+"}image"]
with (folder/"vector_export_receipt_v1.json").open("x",encoding="utf-8") as stream:
    json.dump(report,stream,indent=2)
print(json.dumps({"cropped_vector_PDFs":len(report["PDF_crop"]),"SVG_raster_images_remaining":0,"new_training":0}))
