"""
Evaluation module.
Provides functionality to evaluate generated answers against gold standard answers using an LLM.
"""

from utils import call_llm_api

class Eval:
    """
    Evaluator class that uses an LLM to judge the correctness of an answer.
    """

    def __init__(self):
        """
        Initialize the Eval instance.

        Sets up the LLM client for making evaluation calls.
        """
        self.llm_client = call_llm_api.LLMCompletionCall()
        
    def eval(self, question: str, gold_answer: str, answer: str) -> str:
        """
        Evaluate a predicted answer against a gold answer for a given question.

        Uses the LLM to determine if the predicted answer is semantically correct
        given the gold answer.

        Args:
            question (str): The original question.
            gold_answer (str): The correct (gold standard) answer.
            answer (str): The predicted answer to evaluate.

        Returns:
            str: "1" if the answer is judged correct, "0" otherwise.
        """
        prompt = f"""
        You are an expert evaluator. Your task is to determine if the predicted answer is correct based on the question and gold answer.
        The criteria should be reasonable, not too strict or too lenient.
        
        Question: {question}
        Gold Answer: {gold_answer}
        Predicted Answer: {answer}
        
        Return only "1" (correct) or "0" (incorrect):
        """
        return self.llm_client.call_api(prompt)
