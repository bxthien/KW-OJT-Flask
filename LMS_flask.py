from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import generate_response_with_gemini  # Import the RAG logic

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Invalid request format. 'question' key is required."}), 400

        user_question = data["question"]

        answer = generate_response_with_gemini(user_question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
