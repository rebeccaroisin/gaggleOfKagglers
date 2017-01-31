import pandas as pd

def main():
	data = pd.read_csv("../data/stage1_labels.csv")
	print data.head(10)
	print len(data.index)
	print sum(data["cancer"])

if __name__ == "__main__":
	main()