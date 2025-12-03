from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from utils.config import SEARCH_K
from utils.vector_store import VectorStoreManager
from pathlib import Path


FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

vector_store_manager = VectorStoreManager(True)
app = FastAPI()

# Serve assets (JS/CSS/images)
app.mount(
    "/assets",
    StaticFiles(directory=FRONTEND_DIST / "assets"),
    name="assets",
)

app.mount(
    "/rag.ico",
    StaticFiles(directory=FRONTEND_DIST),
    name="ico",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/{full_path:path}", tags=["root"])
async def read_root(full_path: str):
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"error": "Frontend build not found. Run npm run build."}


@app.post("/search", tags=["search"])
async def search(request: dict) -> dict:
    return { "answer": vector_store_manager.search(request['message'], k=SEARCH_K)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait for text from the client
            data = await websocket.receive_text()
            # Give the answer back
            searchResults = vector_store_manager.search(data, k=SEARCH_K)
            events = [
                {
                    "id": p.id,
                    "score": p.score,
                    "payload": p.payload,
                }
                for p in searchResults
            ]
            await websocket.send_json({ "answer": events})
    except WebSocketDisconnect:
        pass
