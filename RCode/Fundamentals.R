
#Checking basics mathematical codes
func<-function(x) 1/((x+1)*sqrt(x))
integrate(func, lower = 0, upper = Inf)

# Note R is case sensitive and index start with 1

#Data Containers are: Vector, Matrix, Data Frame, List, Environment

#The Vector Object
vec <- c("a",2,23)
print(vec)
vec_2 <- c(1,2,3,4,5,6)
print(vec_2[3])

#The Matrix Object

mat_1 <- matrix(c(1,2,3,4,5,6), nrow = 2,ncol = 3,byrow = TRUE)
mat_1

#Assigning the column names to the function
dimnames(mat_1) <- list(c("one","two"),c("c1","c2","c3"))
mat_1

#Checking the dimensional attribute of the matrix
attributes(mat_1)

#Retrieving the matrix value at specified row and column
ans <- mat_1[1,3]
ans

mat_2 <- sqrt(mat_1)
mat_2

mat_3 <- matrix(rnorm(1000), nrow = 100)
round(mat_3[1:5, 2:6], 3) #Round the values upto 3 decimal place

mat_4 <- mat_3[1:25,]^2 #Slicing some of the values from exisiting matrix
mat_4


#The data.frame object
df <- data.frame(price = c(89.2, 23.2, 21.2),symbol = c("MOT", "AAPL", "IBM"),action = c("Buy", "Sell", "Buy"))
df
class(df) #Print the datatype or object type

print(df[1,2]) #Access the element at particular row and column
print(df[,1])
sum <- df[,1]
sum

""

df2 <-data.frame(col1=c(1,2,3),col2=c(1,2,3,4,5)) #It will show error as the two columns must be of same dimensions
df2

""
#Now moving towards the lists
my_list <- list(a = c(1, 2, 3, 4, 5),
                b = matrix(1:10, nrow = 2, ncol = 5),
                c = data.frame(price = c(79.3, 98.2, 21.2),
                               stock = c("A", "B", "C")))

my_list

first_element <- my_list[[1]]
first_element
length(my_list)



