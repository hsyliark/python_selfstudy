bicycles = ['trek', 'cannondale', 'redline', 'specialized']
print(bicycles)
print(bicycles[0])
print(bicycles[0].title())
print(bicycles[-1])
message = "My first bicycle was a " + bicycles[0].title() + "."
print(message)

motorcycles = ['honda', 'yamaha', 'suzuki']
motorcycles[0] = 'ducati'
print(motorcycles)

motorcycles = ['honda', 'yamaha', 'suzuki']
motorcycles.append('ducati')
print(motorcycles)

motorcycles = ['honda', 'yamaha', 'suzuki']
motorcycles.insert(0, 'ducati')
print(motorcycles)

motorcycles = ['honda', 'yamaha', 'suzuki']
del motorcycles[1]
print(motorcycles)

motorcycles = ['honda', 'yamaha', 'suzuki']
popped_motorcycle = motorcycles.pop()
print(motorcycles)
print(popped_motorcycle)
print("The last motorcycle I owned was a " + popped_motorcycle.title() + ".")

motorcycles = ['honda', 'yamaha', 'suzuki']
first_owned = motorcycles.pop(0)
print("The first motorcycle I owned was a " + first_owned.title() + ".")

motorcycles = ['honda', 'yamaha', 'suzuki', 'ducati']
motorcycles.remove('ducati')
print(motorcycles)

cars = ['bmw', 'audi', 'toyota', 'subaru']
cars.sort()
print(cars)

cars = ['bmw', 'audi', 'toyota', 'subaru']
cars.sort(reverse=True)
print(cars)

cars = ['bmw', 'audi', 'toyota', 'subaru']
print(sorted(cars))

cars = ['bmw', 'audi', 'toyota', 'subaru']
cars.reverse()
print(cars)
len(cars)