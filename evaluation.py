from sys import argv
import pandas as pd

#check column separator, it sould be ","
def load_test_file(path):
	arq = open(path, "r")
	header = arq.readline()
	if(not "user_id" in header.split(",")):
		raise Exception("Column user_id not found")
	lines = arq.readlines()
	arq.close()
	return header, lines

#separator between user_id and items should be a TAB
#separator between items should be a COMMA
#if there is any doubt, please check an example of solution file from 2016 RecSys Challenge
def load_solution_file(path):
	arq = open(path, "r")
	header = arq.readline()
	lines = arq.readlines()
	arq.close()
	return header, lines

def populate_solution_map(path_solution_file):
	solution_lines = load_solution_file(path_solution_file)[1]
	solution_map = {}
	feedback_counter = 0
	n_lines = len(solution_lines)
	for s_line in solution_lines:
		s_line_split = s_line.replace("\n", "").split("\t")
		user = s_line_split[0]
		items = s_line_split[1].split(",")
		if (items[0] != ""):
			solution_map[user] = results = map(int, items)
		#printing taks completion feedback
		feedback_counter = feedback_counter + 1
		if(feedback_counter % ((n_lines + 1) / 4) == 0):
			print ">>>", str(100 * float(feedback_counter) / n_lines).split(".")[0] + "% done"
	print ">>> 100% done"
	print "solution_map populated!"
	return solution_map


#precision within the first top k items
def precisionAtK(recommendedItems, relevantItems, k):
#  if(k > len(recommendedItems)):
#    k = len(recommendedItems)
  topK = recommendedItems[0:k] #takes first k items from the list of reccommendedItems
  return len(set(topK).intersection(set(relevantItems))) / float(k)

#recall = fraction of relevant, retrieved items (30 items 
#are allowed to be submitted at maximum per user): 
def recall(recommendedItems, relevantItems):
  limit = 30
#  if(30 > len(recommendedItems)):
#    limit = len(recommendedItems)
  if (len(relevantItems) > 0):
    return len(set(recommendedItems[0:limit]).intersection(set(relevantItems))) / float(len(relevantItems))
  else:
    return 0.0

#user success = was at least one relevant item recommended for a given user?
def userSuccess(recommendedItems, relevantItems):
  limit = 30
  if(30 > len(recommendedItems)):
    limit = len(recommendedItems)
  if (len(set(recommendedItems[0:limit]).intersection(set(relevantItems))) > 0):
    return 1.0
  else:
    return 0.0

#main scoring function: 
def score(solution_map, tuplas_relevant):
	print "calculating score..."
 	score = 0.0
	for tupla_u_t in tuplas_relevant: #foreach (u, t) in T: #t = set of relevant items for user u
		user = tupla_u_t[0]
		relevant_items = tupla_u_t[1]
		if (str(user) in solution_map.keys()):
			r = solution_map[str(user)] #S(u) #r = ordered list of recommended items for user u
		else:
			r = []
		score = score + (20 * (precisionAtK(r, relevant_items, 2) + precisionAtK(r, relevant_items, 4) + recall(r, relevant_items) + userSuccess(r, relevant_items)) + 10 * (precisionAtK(r, relevant_items, 6) + precisionAtK(r, relevant_items, 20)))
	print "score caltulated!"
	return score


#return a tuple (user, relevantItems) for clicking interactions
def get_relevant_items(path_test_file):
	print "loading test_file data"
	whole_test_data = pd.read_csv(path_test_file)
	print "data loaded!"
	tuplas_relevant = []
	feedback_counter = 0
	users_test_lines = list(whole_test_data.user_id.unique())
	print "processing users_test_data"
	for user in users_test_lines:
		whole_test_data[whole_test_data['user_id'] == user]
		tupla = (user, list(set(whole_test_data[whole_test_data['user_id'] == user].item_id)))
		tuplas_relevant.append(tupla)
		feedback_counter = feedback_counter + 1
		#print n_lines, feedback_counter 
		if(feedback_counter % ((len(users_test_lines) + 1) / 35) == 0):
			print ">>>", str(100 * float(feedback_counter) / len(users_test_lines)).split(".")[0] + "." + str(100 * float(feedback_counter) / len(users_test_lines)).split(".")[1][0:1]+ "% done"
	print ">>> 100% done"
	print "users_test_data processed!"
	return tuplas_relevant


path_test_file = argv[1]
path_solution_file = argv[2]

solution_map = populate_solution_map(path_solution_file)
tuplas_relevant = get_relevant_items(path_test_file)

print "Score:", score(solution_map, tuplas_relevant)


