import os.path, gzip, shutil

directory = './step_0/input'
num_of_files = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

output_file = "./step_0/output/tweets.json"

with open(output_file, 'w') as f:
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

print 'Completed concat'

src = output_file
dest = output_file + '.gz'
print 'Gzip {} to {}'.format(src, dest)

with open(output_file, 'rb') as f_in, gzip.open(dest, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)

print 'Completed gzip'
