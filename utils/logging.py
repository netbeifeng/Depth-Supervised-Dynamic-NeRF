from datetime import datetime

def log(routine, message):
    now = datetime.now() # current date and time
    # datetime in format YYYY-MM-DD HH:MM:SS
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f'{routine.capitalize()} >>> [{date_time}]: {message}')

  