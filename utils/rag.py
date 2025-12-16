from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from utils.config import SEARCH_K
from utils.vector_store import VectorStoreManager
from pathlib import Path


FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist" / "my-angular-app" / "browser"

vector_store_manager = VectorStoreManager(True)
app = FastAPI()

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

# Serve everything in the dist folder as static files
# html=True lets it serve index.html at "/"
app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIST, html=True),
    name="frontend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

