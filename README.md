# Laptop price analysis and prediction
This project is finished with the help of my companion: **Trinh The Hien** supporting the missing data handling task and prediction model

--
# Problem description:
The project goal is to predict the price of a laptop based on its configuration

- Input: 'weight', 'Processor rank', 'Graphics Coprocessor perf', 'Brand', 'Laptop type', 'Laptop purpose', 'Hard Drive Type', 'Memory Type', and 'Operating System'
- Output: Price

Although there are above 70 features of the laptop from the raw data, after many phases of cleaning data, almost all of them just have one value, we choose those features above which have the least null value and the most affected to our target label

# Data collecting
The data is collected from 3 different sources:
- cpubenchmark.net
- notebookcheck.net/Mobile-Graphics-Cards-Benchmark-List.844.0.html
- amazon.com

Amazon website provides laptops and laptop configuration data (****)

The first two websites provide more details about the CPU/GPU name and its ranking/performance data. 

So while I need this, to predict the price of a laptop product, CPU and GPU are definitely such valuable features. But just relying on the amazon data, we only have the string name of it. It means to make the ML model learn these features, we can only use One-hot-encoding, but CPU and GPU have 503 and 191 unique values, respectively. And to both reduce the model complexity and increase the data understanding ability, I extra collect another data source, and then map the string name to an integer value (ranking/performance)

# Data preprocessing
Here are some key steps in this phase:
- Map the string name of CPU/GPU to their Rank/Performance as an integer: Using rule-based and Levenshtein distance to match the raw name and its standard name: 'intel i5 1135 g7' -> 'intel core i5-113g7' -> RANK NUMBER
- Some single features have more than one kind of information will be split: "4 GB DDR4" into "4 GB" for "Memory size" and "DDR4" for "Memory type"
- Transform and remove value having measurement symbol

# EDA
I did some Exploratory data analysis that can be founded in the notebook/EDA.ipynb
![image](https://user-images.githubusercontent.com/84280247/223192302-b3240b7f-492c-4b84-ba9a-90ac9bf8ec52.png)

# Model
Our best-performed model is Random Forest Regressor with R2 score is approximately 0.75


