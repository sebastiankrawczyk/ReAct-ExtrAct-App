import os
import json

class QuestionsManager:
    def __init__(self, queries, storage_path, llm):
        self.queries = queries
        self.count = len(queries)
        self.storage_path = f"{storage_path}/questions.json"
        self.llm = llm
        self.questions = self.load_or_create_questions()

    def generate_questions(self):
        questions = []
        for i, q in enumerate(self.queries):
            topic = q["topic"]
            prompt = f"""
                ## Topic:
                "{topic}"

                ## Create a question:
                Based on the topic above, create a research question directed specifically at the analyzed article. 
                The question should be designed for a RAG system, enabling effective retrieval of information from the article. 
                Focus on vector matching and extracting relevant details.
                Ensure the question is phrased as if addressing the content of the article directly.
                Return the generated question as a plain string without any formatting.
            """
            question = self.llm.complete(prompt)
            questions.append(f"{question!s}")
        return questions
    
    def load_or_create_questions(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r', encoding='utf-8') as file:
                stored_data = json.load(file)
                stored_queries = stored_data.get('queries', [])
                stored_questions = stored_data.get('questions', [])
                
                if stored_queries == self.queries:
                    return stored_questions
                else:
                    new_questions = self.generate_questions()
                    self.save_questions(self.queries, new_questions)
                    return new_questions
        else:
            new_questions = self.generate_questions()
            self.save_questions(self.queries, new_questions)
            return new_questions

    def save_questions(self, queries, questions):
        data = {
            'queries': queries,
            'questions': questions,
        }
        
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        with open(self.storage_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def get_question(self, i):
        return {
            "query": self.queries[i],
            "question": self.questions[i]
        }
    
    def get_questions(self):
        return self.questions
    
    