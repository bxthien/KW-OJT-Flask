import requests
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import re
import google.generativeai as genai
import markdown2

MAX_HISTORY_LENGTH = 5

nltk.download('punkt')
nltk.download('punkt_tab')

SUPABASE_URL = "https://pqqaslokhbmcnhqymfdb.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBxcWFzbG9raGJtY25ocXltZmRiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzYyMjUwNjcsImV4cCI6MjA1MTgwMTA2N30.htMfw6GvvuyJ-oD0sEeLMj_yG3YIEMFwtI1XoE4ZhQg"

GEMINI_API_KEY = "AIzaSyD21doT3y5MEMLlEjR9Wg7-PzP1IS27AEY"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

genai.configure(api_key="AIzaSyD21doT3y5MEMLlEjR9Wg7-PzP1IS27AEY")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")

history = []

introduce_of_table = """
    A brief introduction to the tables is as follows:
    1. users : user_id identifies users. The total number of user_id represents the total number of users.
               user_name refers to the user's name.
               email refers to the user's email.
               created_at indicates the date when the user joined the service.
               is_admin is True if the user is an admin, and False otherwise.
               contact represents the user's phone number.
               date_of_birth represents the user's date of birth in the format YYYY-MM-DD.
               age represents the user's age.
    2. courses : course_id identifies courses. The total number of course_id represents the total number of courses.
                 course_name refers to the name of the course.
                 date_of_update represents the creation date of the course.
                 course_description provides a brief introduction to the course.
                 color represents the color used to display the course in the UI.
    3. chapter : chapter_id identifies chapters. The total number of chapter_id represents the total number of chapters.
                 chapter_name refers to the name of the chapter.
                 quiz_cnt represents the number of quizzes in the chapter.
    4. lecture : lecture_id identifies lectures. The total number of lecture_id represents the total number of lectures.
                 chapter_id is a foreign key referencing chapter_id from the course_detail table.
                 lecture_name refers to the name of the lecture.
                 lecture_document contains the content of the lecture.
    5. user_course_info : This table represents the courses each user is enrolled in.
                          If a tuple exists in this table, it means that the user with user_id is enrolled in the course with course_id.
                          user_id is a foreign key referencing user_id in the users table.
                          course_id is a foreign key referencing course_id in the courses table.
                          status_of_learning indicates the user's course progress: "In Progress" means ongoing, and "Done" means completed.
                          student_enrollment_date represents the date when the user started the course.
    6. user_course_quiz_info : This table provides details about quizzes in the chapters of the courses that a user is enrolled in.
                               It includes information about how many quizzes the user answered correctly out of the total number of quizzes for each chapter.
                               user_id is a foreign key referencing user_id in the users table.
                               course_id is a foreign key referencing course_id in the courses table.
                               chapter_id is a foreign key referencing chapter_id in the course_detail table.
                               correct_answer_cnt indicates how many quizzes the user answered correctly.
    7. course_chapter : This table establishes a Many-to-Many relationship between the courses and chapter tables.
                        course_id is a foreign key referencing course_id in the courses table.
                        chapter_id is a foreign key referencing chapter_id in the chapter table.
"""

def generate_table_selection_prompt(user_question):
    prompt = f"""
        User's Question. The user has asked the following question:
        {user_question}

        Detailed Explanation of Tables:
        {introduce_of_table}

        You need to select the most relevant table(s) from the following list to answer the user's question:
        1. users
        2. courses
        3. chapter
        4. lecture
        5. user_course_info
        6. user_course_quiz_info
        7. course_chapter

        Return only the names of the relevant tables. Exclude unnecessary tables.

        - When providing your answer, include only the necessary table names as shown in the example below.
        ex) If the tables 'users' and 'chapter' are required: => 'users, chapter'

        - If both 'courses' and 'chapter' are selected, also include 'course_chapter'.

        - If 'user_course_quiz_info' is selected, include 'chapter'.

        - If 'user_course_info' is selected, include 'users' and 'courses.'
        
        - If a table contains foreign keys referencing other tables, include the referenced tables.
        ex) When the 'lecture' table needs to be provided, since the 'lecture' table references the 'chapter_id' in the 'chapter' table as a foreign key, the 'chapter' table must also be included.
    """
    return prompt.strip()

def call_gemini_api(prompt):
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{
        "parts":[{"text": prompt}]
        }]
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        print(f"[ERROR] Gemini API call failed. Status code: {response.status_code}")
        print(f"Response content: {response.text}")
        return "Sorry, an error occurred while generating the response."

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    return text

def extract_year(user_input):
    year_match = re.search(r'\b(19|20)\d{2}\b', user_input)
    return int(year_match.group(0)) if year_match else None

def select_relevant_tables(user_question):
    prompt = generate_table_selection_prompt(user_question)
    response = call_gemini_api(prompt)
    tables = [table.strip() for table in response.split(',') if table.strip()]
    print(f"selected table: {tables}")
    return tables

def fetch_data_based_on_question(user_question):
    tables = select_relevant_tables(user_question)
    required_tables = ["users", "courses", "user_course_info", "chapter", "user_course_quiz_info", "lecture", "course_chapter"]
    if not any(table in tables for table in required_tables):
        print("None of the required tables exist. Default tables will be used.")
        tables = required_tables
    return fetch_data_from_tables(tables)

def fetch_data_from_tables(tables, selected_columns=None):
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {}
    if "user_course_info" in tables:
        if "users" in tables:
            tables.remove("users")
        if "courses" in tables:
            tables.remove("courses")

    if "user_course_quiz_info" in tables:
        if "users" in tables:
            tables.remove("users")
        if "courses" in tables:
            tables.remove("courses")
        if "chapter" in tables:
            tables.remove("chapter")

    if "course_chapter" in tables:
        if "courses" in tables:
            tables.remove("courses")
        if "chapter" in tables:
            tables.remove("chapter")

    print(f'Modified tables : {tables}')

    for table in tables:
        query_params = ""
        if table == "user_course_info":
            table = "user_course_view"

        if table == "user_course_quiz_info":
            table = "quiz_view"

        if table == "course_chapter":
            table = "course_chapter_view"

        if table == "lecture":
            table = "lecture_view"

        if selected_columns and table in selected_columns:
            query_params = f"?select={','.join(selected_columns[table])}"
        url = f"{SUPABASE_URL}/rest/v1/{table}{query_params}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data[table] = response.json()
        else:
            print(f"[ERROR] {table} Failed to fetch data from the table. Status code: {response.status_code}")

    return data

PROMPT_TEMPLATE = """
    You are an assistant managing an LMS service, and your name is HOTDOGGY. 
    You are obligated to provide accurate answers to the user's questions. 
    However, you do not need to introduce yourself with every response.

    User's Question:
    {question}

    Please answer the user's question based on the provided context:
    {context}

    Previous questions and generated responses by the user:
    {history_context}

    Detailed explanation of the tables:
    {introduce_of_table}

    
    When generating answers, please refer to {question}, {context}, {history_context}, and {introduce_of_table}, and follow the guidelines below:

    - If {question} is vague, such as "How about course A?" or "What about std2?", refer to {history_context}.  
      If there are questions in {history_context} that have a similar context to {question}, generate a response in a format similar to the previous related response.  
      However, if {question} is determined to be a new, complete sentence that introduces a new context, provide a response regardless of {history_context}.  
      When referring to {history_context}, always start from the most recent user question and the most recent chatbot response.
      ex1) Assume {history_context} contains the following:
          'User: I'm curious about the quiz scores of students taking the FE course.
           Chatbot: std1 - HTML: 9/10 CSS: 8/10
                    std2 - HTML: 7/10 CSS: 5/10

           User: Who is taking the FE course?
           Chatbot: std1, std2 are taking the FE course.'
           If the user then asks, "What about the BE course?", you should refer to the most recent user question, "I'm curious about the quiz scores of students taking the FE course." and its corresponding chatbot response. 
           Interpret the vague question "What about the BE course?" as "Please provide the quiz scores of students taking the BE course" and generate a response accordingly.
      ex2) Assume {history_context} contains the following: 
          'User: Is User1 taking the FE course?
           Chatbot: Yes, User1 is taking the FE course.

           User: I'm curious about the quiz scores of students taking the FE course.
           Chatbot: std1 - HTML: 9/10 CSS: 8/10
                    std2 - HTML: 7/10 CSS: 5/10'
           If the user then asks, "What about User2?", refer to the most recent question, "Is User1 taking the FE course?" and its chatbot response. 
           Interpret the vague question "What about User2?" as "Is User2 taking the FE course?" and generate a response accordingly.

    - Do not include the `id` attribute in your answers. Instead, use the `name` attribute from the same table to provide the answer.
      ex) ''gd35c853-b5me-4f12-94d1-d2339960c3f4' is taking the FE course.' => 'std1 is taking the FE course.'
         In the above example, if `gd35c853-b5me-4f12-94d1-d2339960c3f4` is the `user_id`, replace it with the `user_name` in your response.

    - Always provide concise and accurate answers. Ensure your response is polite and respectful.

    - Even if there is only limited data available, use the available data to provide a precise response. Do not generate responses indicating insufficient data.  

    - If you cannot provide an answer, do not provide false information. Instead, reply with, "I'm sorry, I cannot provide an answer."  

    - If the question does not explicitly include details about the user's quiz information, enrolled courses, or other specific user details, do not include user-related information in your response.
      ex) 'Explain the FE course'
           => Provide a brief overview of the chapters and description of the FE course.

          'Explain the FE course and provide information about the users taking it'
           => Provide both the course overview and information about the users (excluding the id attribute).

    - Consider `nick_name` and `nickname` as equivalent. Do not worry about underscores in attribute names.
    

    1. If {question} contains a year and asks about trends or popular items during that year, check attributes with a `timestamp` type.  
       Identify data where the year in the `timestamp` matches the year in {question}, and the most frequently occurring data represents the trend for that year.

    2. If `correct_answer_cnt` is less than `(quiz_cnt * 0.8)`, it means that the chapter's learning progress is insufficient.  
       If the user asks about quiz scores for specific chapters, display them in the format "correct/total questions" and provide feedback or a review for insufficient chapters.  
       If `correct_answer_cnt` does not exist, indicate that the quizzes have not been attempted yet (e.g., "Not attempted").

    3. If a question requests additional examples or feedback, but the document does not contain detailed information or examples, provide appropriate examples or feedback independently.

    4. If the question is unrelated to the LMS service, respond with: "This question is unrelated to the LMS service."
       ex) 'What should I have for lunch?' => "This question is unrelated to the LMS service."

    5. For {context}:  
        - If the table name is `users`, it represents users. The total count of `user_id` is the total number of users.  
        - If the table name is `courses`, it represents courses. The total count of `course_id` is the total number of courses.  
        - If the table name is `chapter`, it represents chapters. The total count of `chapter_id` is the total number of chapters.  
        - If the table name is `lecture_view`, it represents lectures. The total count of `lecture_id` is the total number of lectures. 

    6. If {question} is in the form "Who is taking course A?", check the table named `user_course_view`.  
       Identify data where `course_name` matches the course name in {question} and provide the corresponding `user_name`.

    7. If {question} is in the form "What courses is std1 taking?", check the table named `user_course_view`.  
       Identify data where `user_name` matches the user's name in {question} and provide the corresponding `course_name`.  

    8. If {question} asks about the number of quizzes answered correctly for specific courses or chapters, check the table named `quiz_view`.  
       Identify data where `course_name` or `chapter_name` matches the names in {question}, and use `correct_answer_cnt` for correct answers and `quiz_cnt` for the total number of questions.

    9. If {question} is about the relationship between courses and chapters, refer to the table named `course_chapter_view`.

    10. If the details of a lecture are required, include the response: 'Please refer directly to the lecture for detailed content.'
"""

def create_prompt(user_input, relevant_data, history):
    if not relevant_data:
        return f"I'm sorry, no data related to '{user_input}' could be found."

    relevant_docs_content = "\n".join(
        f"{i + 1}. table: {table}, data: {row}" for i, (table, row, _) in enumerate(relevant_data)
    )

    print(relevant_docs_content)
    prompt = PROMPT_TEMPLATE.format(
        question=user_input,
        context=relevant_docs_content,
        history_context = history,
        introduce_of_table = introduce_of_table
    )

    return prompt.strip()

def interpret_user_question(user_question, history):
    if not history:
        return user_question

    last_question, _ = history[-1]
    matched_course = re.search(r'(\w+)-course', last_question, re.IGNORECASE)
    if matched_course:
        last_course = matched_course.group(1).upper()
        if "course" not in user_question.lower():
            return f"{last_course}-Is it a question about the course? {user_question}"
    return user_question

def find_relevant_data(user_input, data):
    print(user_input)
    tokenized_user_input = word_tokenize(preprocess_text(user_input))
    print(tokenized_user_input)
    year = extract_year(user_input)
    relevant_data = []

    for table, rows in data.items():
        documents = [preprocess_text(" ".join([str(value) for value in row.values()])) for row in rows]
        tokenized_documents = [word_tokenize(doc) for doc in documents]

        bm25 = BM25Okapi(tokenized_documents)
        scores = bm25.get_scores(tokenized_user_input)

        for i, score in enumerate(scores):
            if year:
                timestamp_field = next((field for field in rows[i] if isinstance(rows[i][field], str) and "T" in rows[i][field]), None)
                if timestamp_field and str(year) in rows[i][timestamp_field]:
                    relevant_data.append((table, rows[i], score))
            else:
                relevant_data.append((table, rows[i], score))

    return sorted(relevant_data, key=lambda x: x[2], reverse=True)

def generate_response_with_gemini(user_question):
    global history

    history_context = "\n".join(
        [f"User: {q}\nChatbot: {a}" for q, a in reversed(history)]
    )
    if history_context:
        full_prompt = f"""
        Previous conversation:
        {history_context}

        New question:
        User: {user_question}
        """
    else:
        full_prompt = user_question
    
    print("Current history:")
    print(history_context)
    data = fetch_data_based_on_question(user_question)
    relevant_data = find_relevant_data(user_question, data)

    context_prompt = create_prompt(user_question, relevant_data, history_context)

    final_prompt = f"{full_prompt}\n\n{context_prompt}"
    response = call_gemini_api(final_prompt)

    html_response = markdown2.markdown(response)

    history.append((user_question, response))
    if len(history) > MAX_HISTORY_LENGTH:
        history.pop(0)

    return html_response
