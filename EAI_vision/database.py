class Credentials:
    def __init__(self, database_path):
        self.path = database_path
        self.database = self.get_database()

    def get_database(self):
        database = open(self.path)
        content = database.read()
        data_list = content.split('\n')
        return data_list

    def write_database(self):
        user_id = input(f'Please choose your user ID: ').lower()
        while True:
            if not user_id:
                print(f'Input is not in required format! Please try again!')
                user_id = input(f'Please choose your user ID: ').lower()
            else:
                if user_id in self.database:
                    print('This user ID is unavailable, please try again!')
                    user_id = input(f'Please choose your user ID: ').lower()
                else:
                    database = open(self.path, 'a')
                    database.write(user_id + '\n')
                    database.close()
                    return user_id

    def check_reg(self, answer):
        result = 0
        while answer is not 'yes' and answer is not 'and':
            answer = 'Please write yes or no'
            if answer is 'yes':
                print(f'You are directed to the registration phase!')
                result = 1
                return result
            if answer is 'no':
                print(f'You will have only 1 chance left. Then program flow will be terminated!')
                result = 2
                return result

    def get_Username(self):
        user_id = input(f'Please enter your user_id: ').lower()
        result = 0
        counter = 3
        while True:
            if not user_id:
                print(f'Input is not in required format! Please try again!')
                user_id = input(f'Please enter your user_id: ').lower()

            else:
                if user_id not in self.database:
                    print('This user_id was not registered, please try again!')
                    user_id = input(f'Please enter your user_id: ').lower()
                    if counter == 0:
                        ans = input('Dear guest! Apparently, you are not registered. Do you want to register?')
                        self.check_reg(ans)
                    counter = counter -1


                else:
                    return user_id



# database_path = "Database/usernames.txt"
# credentials = Credentials(database_path)
# username = credentials.get_Username()
# print(f'Welcome {username}')
