from __future__ import annotations
import hashlib,json,re
from pathlib import Path

HAND_DRAWN="hand_drawn"; HAND_IMPORTED="hand_drawn_on_imported_task"; RETOUCHED="sam2_retouched"; EDITED="sam2_edited"; ACCEPTED="sam2_accepted"; TRANSFERRED="transferred"
UPLOADED=re.compile(r"(video_0[1-4])_f(\d{4})\.png$")

def digest(value):return hashlib.sha1(value.encode("utf-8")).hexdigest()
def classify(origins,source):
    origins=set(origins)
    if source is None:return HAND_DRAWN
    if origins=={"manual"}:return HAND_IMPORTED
    if source.get("import_source")=="transferred_from_day_for_review":return TRANSFERRED
    if "prediction-changed" in origins:return EDITED
    if origins=={"prediction","manual"}:return RETOUCHED
    if origins=={"prediction"}:return ACCEPTED
    raise ValueError(f"orígenes no soportados: {sorted(origins)}")

def build_provenance(exports, imports, first_export=None):
    imported={}
    for path in imports:
        for task in json.loads(Path(path).read_text(encoding="utf-8")):
            data=task.get("data") or {}; image=data.get("image")
            if isinstance(image,str) and image.startswith("data:image"):
                imported[digest(image)]={"video":Path(data["source_video"]).stem,"frame":int(data["frame_index"]),"import_source":data.get("annotation_source"),"source_anchor_frame":data.get("source_anchor_frame")}
    lead={}
    if first_export and Path(first_export).exists():
        for task in json.loads(Path(first_export).read_text(encoding="utf-8")):
            image=next(iter((task.get("data") or {}).values()),"")
            for ann in task.get("annotations") or []:
                if ann.get("lead_time") is not None:lead[digest(image)]=float(ann["lead_time"])
    result=[]
    for video,path in exports.items():
        for task in json.loads(Path(path).read_text(encoding="utf-8")):
            anns=[a for a in task.get("annotations") or [] if a.get("result") and not a.get("was_cancelled")]
            if not anns:continue
            ann=max(anns,key=lambda x:str(x.get("updated_at") or ""));image=next(iter(task.get("data",{}).values()))
            match=UPLOADED.search(image)
            if match:source=None;tv=match.group(1);frame=int(match.group(2))
            else:
                source=imported.get(digest(image))
                if source is None:continue
                tv=source["video"];frame=source["frame"]
            if tv!=video:continue
            origins={r.get("origin") for r in ann["result"]}
            result.append({"video":video,"frame_index":frame,"task_id":task.get("id"),"annotation_id":ann.get("id"),"updated_at":ann.get("updated_at"),"lead_time_s":lead.get(digest(image)),"result_origins":sorted(origins),"category":classify(origins,source),"source_anchor_frame":None if source is None else source.get("source_anchor_frame"),"import_source":"uploaded_image" if source is None else source.get("import_source")})
    return sorted(result,key=lambda r:(r["video"],r["frame_index"]))

def validate_anchors(rows,boundaries_by_video):
    redraw=[]
    for row in rows:
        bounds=boundaries_by_video[row["video"]]
        row["segment_id"]=sum(row["frame_index"]>=b for b in bounds)-1;row["valid"]=True;row["exclusion_reason"]=None
        source=row.get("source_anchor_frame")
        if source is not None and sum(source>=b for b in bounds)-1 != row["segment_id"]:
            row["valid"]=False;row["exclusion_reason"]="source_anchor_en_otro_segmento";redraw.append(dict(row))
    return rows,redraw
