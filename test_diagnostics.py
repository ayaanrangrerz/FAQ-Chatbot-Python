from Chatbot import _USE_SKLEARN, vectorizer, vectors, doc_vectors, _idf, questions, faq, _tokenize, _cosine_sim

queries = [
    'What is AI?',
    'Tell me about AI',
    'What is Python?',
    'Define Python',
    'What is Machine Learning?',
    'Explain machine learning'
]

for q in queries:
    print('\nQuery:', q)
    if _USE_SKLEARN:
        qv = vectorizer.transform([q])
        sims = list(__import__('sklearn.metrics.pairwise').metrics.pairwise.cosine_similarity(qv, vectors)[0])
        print('sims:', sims)
        idx = int(max(range(len(sims)), key=lambda i: sims[i]))
        print('chosen:', idx, questions[idx], '->', faq[questions[idx]])
    else:
        toks = _tokenize(q)
        qset = set([t for t in toks if t not in {"what","is","the","a","an","of","in","to"}])
        overlaps = []
        for i, qq in enumerate(questions):
            q_tokens = set([t for t in _tokenize(qq) if t not in {"what","is","the","a","an","of","in","to"}])
            overlaps.append(len(qset & q_tokens))
        print('overlaps:', overlaps)
        # TF-IDF fallback
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        qvec = {t: (tf[t] * _idf.get(t, 1.0)) for t in tf}
        norm = sum(v*v for v in qvec.values())**0.5
        if norm>0:
            for t in qvec:
                qvec[t]/=norm
        sims = [_cosine_sim(qvec, dv) for dv in doc_vectors]
        print('tfidf sims:', sims)
        idx = int(max(range(len(sims)), key=lambda i: sims[i]))
        print('chosen:', idx, questions[idx], '->', faq[questions[idx]])
