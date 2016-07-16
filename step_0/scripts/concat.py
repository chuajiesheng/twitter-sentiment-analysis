import os, os.path

directory = './step_0/input'
num_of_files = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

output_file = "./step_0/output/tweets.json"
f = open(output_file, 'w')

for x in range(0, num_of_files):
	filename = '{}/{}.json'.format(directory, x)
	with open(filename, 'r') as f2:
		print 'Reading {}'.format(filename)
		for line in f2:
			if len(line.strip()) < 1:
				continue
			f.write(line.strip())
			f.write('\n')
			f.flush()

f.close()