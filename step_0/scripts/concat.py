import os, os.path

directory = './step_0/input'
num_of_files = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

output_file = "./step_0/output/tweets.json"
with open(output_file, 'w') as f:
	for x in range(0, num_of_files):
		filename = '{}/{}.json'.format(directory, x)
		with open(filename) as j:
			print 'Reading {}'.format(filename)
			for line in j:
				if len(line.strip()) < 1:
					continue
                f.write(line.strip())
                f.write('\n')
