
def main():
    mileage = float(input("Enter a mileage: "))
    with open("theta", 'r') as file:
        theta = file.read().strip().split(',')
    slope = float(theta[1])
    intercept = float(theta[0])
    print("Predicted price is: ", (slope * mileage) + intercept)

if __name__ == '__main__':
    main()
