# Laptop price analysis and prediction
This project is finished with the help from my companion: **Trinh The Hien** supporting for the missing data handling task and prediction model

--
# Problem description:
The project goal is to predict the price of a laptop basing on its configuration

- Input: 'weight','Processor rank', 'Graphics Coprocessor perf', 'Brand', 'Laptop type', 'Laptop purpose', 'Hard Drive Type', 'Memory Type' and 'Operating System'
- Output: Price

Although there are above 70 features of laptop from the raw data, but after many phases of cleaning data, almost all of them just have one value, we choose those features above which have the least null value and the most affected to our target label

# Data collecing
The data is collected from 3 different sources:
- cpubenchmark.net
- notebookcheck.net/Mobile-Graphics-Cards-Benchmark-List.844.0.html
- amazon.com

Amazon website provides laptops and laptop configuration data (****)

The first two websites provide more details about the CPU/GPU name and its ranking/performance data. 

So while I need this, to predict price of a laptop product, CPU and GPU is definitely such the valueable features. But just relying on the amazon data, we only have the string name of it. It means to make the ML model learning these features, we can only use One-hot-encoding, but CPU and GPU have 503 and 191 unique values, respectively. And to either redude the model complexity and increase the data understanding ability, I extra collect another data source, and then map the string name to an integer value (ranking/performance)
