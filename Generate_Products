def generate_product(id):
    delete_file_contents(filename)
    p = []
    p = find_products(id)
    ms=set()
    for i in p:
      ms.add(str(i))

    #p = set((p))  # Remove duplicates while maintaining order
    print("Recommended Categories:", ms)

    prod = []
    s=set()
    for category in p:
      for value in category:
        category_products = ab.loc[ab['Category'] == value, 'Product_Name']

        random_values = category_products.sample(n=5, replace=True)  # Sample with replacement to ensure same number of products
        for product in random_values:
            prod.append(product)
            s.add(product)
    # print("Recommended Products:", prod)

    my_list = list(s)

# Create a DataFrame from the list
    d = pd.DataFrame(my_list, columns=['RProducts'])

# Save the DataFrame to a file (e.g., CSV)
    d.to_csv( '/content/drive/MyDrive/Colab Notebooks/Capstone_Project/static/ProductRecommendations.csv', index=False)

    return s



id=8387 #Sample Customer ID
product_names = generate_product(id)
