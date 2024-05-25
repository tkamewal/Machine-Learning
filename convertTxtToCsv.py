import pandas as pd

# Replace 'input_file.data' with the path to your .data file
input_file_path = 'Not.txt'

# Replace 'output_file.csv' with the desired name for the output .csv file
output_file_path = 'LinearTest.csv'

df = pd.read_csv(input_file_path, delimiter=',')

# Save the DataFrame to a CSV file
df.to_csv(output_file_path, index=False)

print(f'Conversion from {input_file_path} to {output_file_path} completed.')
