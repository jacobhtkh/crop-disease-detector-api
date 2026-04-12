from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()
UPLOAD_DIR = Path("uploads")


class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.post("/upload")
async def upload_image(files: List[UploadFile] = File()):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for file in files:
        contents = await file.read()
        name = Path(file.filename or "upload.bin").name
        dest = UPLOAD_DIR / name
        with dest.open("wb") as f:
            f.write(contents)
        saved.append(name)
    return {"filenames": saved}