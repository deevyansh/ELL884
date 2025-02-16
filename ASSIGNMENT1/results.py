# import matplotlib.pyplot as plt
# import re
# from ngram import *

# # Prepare to store perplexity scores for different n-grams
# n_values = []
# perplexity_scores = []

# # Loop through different n-gram values (1 to 3 in your case)
# for i in range(1, 5):
#     print(i)
#     n_gram = NGramBase()
#     n_gram.n = i
    
#     # Read the data from train1.txt and train2.txt
#     with open("data/train1.txt", "r") as file:
#         test_sentence = file.read()
#     with open("data/train2.txt", "r") as file:
#         test_sentence += file.read()

#     # Fit the n-gram model on the sentences
#     n_gram.fit(n_gram.doit(test_sentence))

#     # Preprocess the test sentence
#     test_sentence = test_sentence.lower()
#     test_sentence = re.sub(r'[^.\w\s]', '', test_sentence)
#     sentences = test_sentence.split('.')
    
#     # Compute perplexity for each sentence and average them
#     score = 0
#     for sentence in sentences:
#         score += n_gram.perplexity(sentence)
#     score = score / len(sentences)
    
#     # Store n and its corresponding perplexity score
#     n_values.append(i)
#     perplexity_scores.append(score)

# # Plot the results using matplotlib
# plt.plot(n_values, perplexity_scores, marker='o')
# plt.xlabel('N-gram order (N)')
# plt.ylabel('Perplexity per sentence')
# plt.title('Perplexity vs N-gram Order')
# plt.xticks(n_values)  # Ensure x-axis ticks correspond to the n-gram values
# plt.grid(True)
# plt.show()



# import matplotlib.pyplot as plt
# from smoothing_classes import *
# import re


# def doit(n_gram):
#     # Read the data from train1.txt and train2.txt
#     with open("data/train1.txt", "r") as file:
#         test_sentence = file.read()
#     with open("data/train2.txt", "r") as file:
#         test_sentence += file.read()

#     # Fit the n-gram model on the sentences
#     n_gram.fit(n_gram.doit(test_sentence))

#     # Preprocess the test sentence
#     test_sentence = test_sentence.lower()
#     test_sentence = re.sub(r'[^.\w\s]', '', test_sentence)
#     sentences = test_sentence.split('.')

#     # Compute perplexity for each sentence and average them
#     score = 0
#     for sentence in sentences:
#         score += n_gram.perplexity(sentence)
#     score = score / len(sentences)
#     return score


# L = []
# smoothing_methods = ['AddK', 'StupidBackoff', 'GoodTuring', 'Interpolation', 'KneserNey']
# scores = []

# # Test each smoothing method
# n_gram = AddK(0.1)
# scores.append(doit(n_gram))

# n_gram = StupidBackoff()
# scores.append(doit(n_gram))

# n_gram = GoodTuring()
# scores.append(doit(n_gram))

# n_gram = Interpolation()
# scores.append(doit(n_gram))

# n_gram = KneserNey()
# scores.append(doit(n_gram))

# # Create a bar graph for the results
# plt.figure(figsize=(10, 6))
# plt.bar(smoothing_methods, scores, color='skyblue')
# plt.xlabel('Smoothing Methods')
# plt.ylabel('Perplexity')
# plt.title('Perplexity Scores for Different Smoothing Techniques')
# plt.show()


# l = [0.001, 0.01, 0.1, 0.2, 0.5]
# scores = []

# # Test each value of l for the AddK smoothing method
# for i in range(len(l)):
#     ngram = AddK(l[i])
#     scores.append(doit(ngram))

# # Create a line graph for the results
# plt.figure(figsize=(10, 6))
# plt.plot(l, scores, marker='o', color='b', linestyle='-', markersize=8)
# plt.xlabel('Value of K (l)', fontsize=12)
# plt.ylabel('Perplexity', fontsize=12)
# plt.title('Perplexity vs K (AddK Smoothing)', fontsize=14)
# plt.grid(True)
# plt.show()


# def doit1(n_gram):
#     # Read the data from train1.txt and train2.txt
    
#     with open("data/train2.txt", "r") as file:
#         test_sentence = file.read()

#     # Fit the n-gram model on the sentences
#     n_gram.fit(n_gram.doit(test_sentence))

#     with open("data/train1.txt", "r") as file:
#         test_sentence = file.read()
#     test_sentence = test_sentence.lower()
#     test_sentence = re.sub(r'[^.\w\s]', '', test_sentence)
#     sentences = test_sentence.split('.')

#     # Compute perplexity for each sentence and average them
#     score = 0
#     for sentence in sentences:
#         score += n_gram.perplexity(sentence)
#     score = score / len(sentences)
#     return score




# # Test each value of l for the AddK smoothing method
# l=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# scores=[]
# for i in range(len(l)):
#     ngram = StupidBackoff(l[i])
#     print("done")
#     scores.append(doit1(ngram))

# # Create a line graph for the results
# plt.figure(figsize=(10, 6))
# plt.plot(l, scores, marker='o', color='b', linestyle='-', markersize=8)
# plt.xlabel('Value of alpha', fontsize=12)
# plt.ylabel('Perplexity', fontsize=12)
# plt.title('Perplexity vs alpha (stupid backoff)', fontsize=14)
# plt.grid(True)
# plt.show()

# Create a grid of values for i and j
# i_values = np.arange(0.2, 0.4, 0.1)
# j_values = np.arange(0.2, 0.4, 0.1)

# Initialize an empty list to store the perplexity scores
# scores = []

# Test different combinations of i and j for the Interpolation smoothing method
# for i in i_values:
#     for j in j_values:
#         l = [0,i, j, 1 - (i + j)]
#         ngram = Interpolation(l)
#         scores.append(doit1(ngram))

# Reshape scores to match the grid
# scores = np.array(scores).reshape(len(i_values), len(j_values))

# Create a contour plot to visualize the results
# plt.figure(figsize=(10, 6))
# cp = plt.contourf(i_values, j_values, scores, cmap='viridis')
# plt.colorbar(cp)
# plt.xlabel('Value of unigram lambda', fontsize=12)
# plt.ylabel('Value of bigram lambda', fontsize=12)
# plt.title('Perplexity vs. unigram lambda and bigram lambda (Interpolation Smoothing)', fontsize=14)
# plt.show()


