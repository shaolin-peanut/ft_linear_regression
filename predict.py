def predict(slope, intercept, mileage):
    return (slope * mileage) + intercept

def main():
    mileage = float(input("Enter a mileage: "))
    with open("theta", 'r') as file:
        theta = file.read().strip().split(',')
    prediction = predict(float((theta[1])), float(theta[0]), mileage);
    print("Predicted price is: ", prediction);

if __name__ == '__main__':
    main()
