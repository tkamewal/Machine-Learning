sales = [
    ("product1", 100),
    ("product2", 150),
    ("product1", 200),
    ("product3", 50),
]
mapped_data = [(product, amount) for product, amount in sales]


reduced_data = {}
for product, amount in mapped_data:
    if product not in reduced_data:
        reduced_data[product] = amount
    else:
        reduced_data[product] += amount

total = sum(reduced_data.values())

print("Total Revenue:", total)
