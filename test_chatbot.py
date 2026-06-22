from Chatbot import _most_similar, questions, faq
print(faq[questions[_most_similar('What is AI?')]])
print(faq[questions[_most_similar('What is Python?')]])
print(faq[questions[_most_similar('What is Machine Learning?')]])
