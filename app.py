import os
import glob
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import MultiAgentSystem

load_dotenv()
app = FastAPI(title="JavaScript Chat")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    rag = MultiAgentSystem()
except Exception as e:
    print("RAG initiation failed", e)
    rag = None



class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    def get_next_history_number(directory: str = ".") -> int:
        """Находит следующий доступный номер для файла истории."""
        # ищем все файлы, соответствующие шаблону history-*.json
        pattern = os.path.join(directory, "history-*.json")
        existing_files = glob.glob(pattern)
        
        if not existing_files:
            return 1
        
        numbers = []
        for file_path in existing_files:
            filename = os.path.basename(file_path)
            try:
                # извлекаем число между "history-" и ".json"
                num_str = filename.replace("history-", "").replace(".json", "")
                numbers.append(int(num_str))
            except ValueError:
                continue
                
        return max(numbers) + 1 if numbers else 1

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл index.html не найден")


@app.post("/chat")
def chat(request: ChatRequest):
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG is down")

    try:
        last_user_message = None
        for msg in reversed(request.messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break

        if not last_user_message:
            raise HTTPException(status_code=400, detail="No user message")

        answer = rag.ask_with_history(request.messages, last_user_message)
         
        # --- логика сохранения истории ---
        next_num = ChatRequest.get_next_history_number()
        filename = f"history-{next_num}.json"
        
        # сохраняем всю историю сообщений текущей сессии
        # можно сохранить только request.messages, или добавить метаданные (время, ответ и т.д.)
        data_to_save = {
            "session_id": next_num,
            "messages": request.messages,
            "last_answer": answer
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            
        print(f"История сохранена в файл: {filename}")
        # -------------------------------

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")