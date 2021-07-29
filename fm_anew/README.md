author: guguowei,2021/07/29, guoweigu@outlook.com  

不使用tensorflow和pytorch等高级类库实现 Factorization Machines 算法  

FM算法的解析很多，我就不用解释了，我代码和注释还是非常清晰的    

sigmod  函数这里可以采用google的word2vec源码中的方式：分区间计算出来，直接取值即可  

还需要做的内容：
1. 使用ftrl算法优化  
2. 提高程序的扩展性、增加实际运行过程中需要的部分  
3. 寻找更好的找到更好超参数的方法  
