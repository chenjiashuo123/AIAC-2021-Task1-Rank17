from operator import itemgetter # 导入定位的头方便定位按照哪里排序
import csv

if __name__ == '__main__':
    datas = [] # 开个列表存放排序过的数据
    with open('data/pairwise/label.tsv','r') as f:
        table=[]
        for line in f:
            col = line.split('\t')
            col[2] = col[2].strip("\n")
            table.append(col)
        table_sorted = sorted(table,key=itemgetter(0),reverse=False) #精确的按照第1列排序
        for row in table_sorted:
            datas.append(row)
    f.close()

    with open("sort_label.tsv","w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for data in datas:
            print(data)
            writer.writerow(data)
    csvfile.close()
