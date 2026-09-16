import sys
import json
import logging
import traceback
import os

# Redirection de la sortie standard vers stderr pour protéger le canal JSON-RPC
original_stdout = sys.stdout
sys.stdout = sys.stderr
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

# Désactivation des warnings HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ECC_Conscience"

try:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    client = QdrantClient(url=QDRANT_URL)
except Exception as e:
    sys.stderr.write(f"Error initializing: {e}\n")

# Restauration de la sortie standard pour JSON-RPC
sys.stdout = original_stdout

def handle_request(request):
    req_id = request.get("id")
    method = request.get("method")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "hephaistos-memory", "version": "1.0.0"}
            }
        }
    
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "search_ecc_conscience",
                        "description": "Recherche sémantique multilingue dans la conscience (RAG) du projet ECC. Utilisez cet outil pour trouver des règles, des workflows, des skills ou des informations du manuel.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "La question ou le terme de recherche (en français ou anglais)."
                                }
                            },
                            "required": ["query"]
                        }
                    }
                ]
            }
        }
        
    elif method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if tool_name == "search_ecc_conscience":
            query = args.get("query", "")
            try:
                vector = model.encode(query, show_progress_bar=False).tolist()
                
                try:
                    response = client.query_points(
                        collection_name=COLLECTION_NAME,
                        query=vector,
                        limit=3
                    )
                    results = response.points
                except AttributeError:
                    results = client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=vector,
                        limit=3
                    )
                
                content = []
                for res in results:
                    score = res.score
                    payload = res.payload
                    filename = payload.get("filename", "Inconnu")
                    domain = payload.get("domain", "Inconnu")
                    text = payload.get("text", "")
                    content.append(f"--- Fichier: {filename} (Domaine: {domain}, Score: {score:.4f}) ---\n{text}\n")
                
                final_text = "\n".join(content) if content else "Aucun résultat trouvé."
                
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": final_text}]
                    }
                }
            except Exception as e:
                sys.stderr.write(f"Search error: {e}\n{traceback.format_exc()}\n")
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Erreur lors de la recherche: {str(e)}"}]
                    }
                }
                
    elif method == "notifications/initialized":
        return None
        
    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }
    return None

def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
