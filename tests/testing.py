import lingutok
lingutok.load()

testcases = []
desired_result = []
result = []
failed_count = 0

with open("testcase.txt", "r") as f:
	for line in f:
		t = line.split("\t")
		testcases.append(t[0])
		desired_result.append(t[1][:-1])

	print("Testing {} cases...\n".format(len(testcases)))
	
	digit = str(len(str(len(testcases))))
	tc_len = str(len(max(testcases)))
	dr_len = str(len(max(desired_result)))
	for word in testcases:
		result.append(str(lingutok.tokenize(word)))
	r_len = str(len(max(result)))

	for i in range(len(result)):
		if (result[i] != desired_result[i]):
			s = "Failed at case no.{:>"+digit+"}, {:<"+tc_len+"}| Result: {:<"+r_len+"}| Desired result: {:<"+dr_len+"}"
			print(s.format(i, testcases[i], result[i], desired_result[i]))
			failed_count += 1

print("\nTotal number of failure: {}".format(failed_count))
print("Successful rate: {:.2f}%".format(((len(testcases)-failed_count)/len(testcases))*100))