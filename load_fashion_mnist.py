import os
k=0
a=33
while k!=2:
	if os.stat("file.txt").st_size==0:
		print(k)
		k+=1
		os.system('python fashion_mnist.py')
	else:
		file=open("file.txt",'r')
		input1=float(file.read())
		file.close()
		if input1>=80.00:
			k==False
			print("Sucess")
			break
		else:
			print(k)
			f=open("fashion_mnist.py",'r')
			reader=f.readlines()
			if k<=1:
				reader[a]="    model.add(Conv2D(15, (3, 3), activation='relu'))\n    model.add(MaxPooling2D())\n\n"
				a+=2
			else:
				reader[a]="    model.add(Conv2D(15, (3, 3), activation='relu'))\n    model.add(MaxPooling2D())\n\n"
			f.close()
			f=open("fashion_mnist.py",'w')
			f.writelines(reader)
			f.close()
			os.system('python fashion_mnist.py')

