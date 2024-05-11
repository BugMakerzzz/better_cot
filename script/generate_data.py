import jsonlines
import random

random.seed(17)

def generate_name(n_samples):

    first_names = ['John', 'Emma', 'Michael', 'Olivia', 'William', 'Ava', 'James', 'Sophia', 'Benjamin', 'Isabella', 'Daniel', 'Mia', 'Joseph', 'Charlotte', 'David', 'Amelia', 'Alexander', 'Harper', 'Matthew', 'Evelyn', 'Andrew', 'Abigail', 'Christopher', 'Emily', 'Joshua', 'Elizabeth', 'Samantha', 'Nicholas', 'Madison', 'Jackson', 'Avery', 'Ryan', 'Chloe', 'Samuel', 'Ella', 'Jonathan', 'Grace', 'Nathan', 'Victoria', 'Christian', 'Scarlett', 'Tyler', 'Hannah', 'Dylan', 'Aria', 'Caleb', 'Lily', 'Mason', 'Zoe', 'Ethan', 'Penelope', 'Henry', 'Layla', 'Nora', 'Sebastian', 'Riley', 'Jack', 'Leah', 'Owen', 'Audrey', 'Wyatt', 'Stella', 'Luke', 'Ellie', 'Julian', 'Paisley', 'Levi', 'Skylar', 'Gabriel', 'Violet', 'Isaac', 'Claire', 'Lincoln', 'Brooklyn', 'Landon', 'Savannah', 'Carter', 'Genesis', 'Jayden', 'Bella', 'Naomi', 'Elena', 'Alexa', 'Serenity', 'Ariana', 'Maya', 'Valentina', 'Alice', 'Gabriella', 'Julia', 'Taylor', 'Aubrey', 'Autumn', 'Faith', 'Hazel', 'Hailey', 'Genesis', 'Piper', 'Willow', 'Eva', 'Quinn', 'Sadie', 'Nevaeh', 'Brielle', 'Peyton', 'Ruby', 'Sophie', 'Natalie', 'Serena', 'Luna', 'Vivian', 'Eleanor', 'Gianna', 'Isla', 'Clara', 'Lydia', 'Delilah', 'Alexandra', 'Elise', 'Hadley', 'Mackenzie', 'Kaylee', 'Sara', 'Jasmine', 'Malia', 'Adeline', 'Eliana', 'Charlie', 'Nadia', 'Juliette', 'Maria']
    last_names = ['Smith', 'Johnson', 'Brown', 'Taylor', 'Miller', 'Anderson', 'Wilson', 'Thomas', 'Lee', 'Harris', 'Clark', 'Lewis', 'Young', 'Walker', 'Hall', 'Allen', 'King', 'Baker', 'Green', 'Adams', 'Scott', 'Mitchell', 'Roberts', 'Carter', 'Phillips', 'Evans', 'Turner', 'Parker', 'Collins', 'Edwards', 'Stewart', 'Morris', 'Cook', 'Rogers', 'Murphy', 'Bell', 'Ward', 'Bailey', 'Cooper', 'Richardson', 'Cox', 'Howard', 'Ward', 'Coleman', 'Kelly', 'Rivera', 'Peterson', 'Gonzalez', 'Reed', 'Cruz', 'Hughes', 'Washington', 'Butler', 'Simmons', 'Foster', 'Gomez', 'Perry', 'Long', 'Patterson', 'Barnes', 'Ross', 'Henderson', 'Cole', 'Jenkins', 'Bryant', 'Morgan', 'Brady', 'Simpson', 'Holt', 'Arnold', 'Crawford', 'Gardner', 'Lopez', 'Perez', 'Torres', 'Hill', 'Ray', 'Rice', 'Riley', 'Bishop', 'Knight', 'Reyes', 'Hansen', 'Porter', 'Hicks', 'Boyd', 'Mason', 'Ross', 'Wells', 'Fisher', 'Russell', 'Griffin', 'Murray', 'Barnett', 'Banks', 'Woods', 'Coleman', 'Warren', 'Johnston', 'Porter', 'Thomas', 'Pierce', 'Hayes', 'Harrington', 'Dixon', 'Hudson', 'Bryant', 'Hunt', 'Harrison', 'Cunningham', 'Black', 'Holmes', 'Duncan', 'Ryan', 'Gardner']
    random_names = random.sample(list(zip(first_names, last_names)), n_samples)

    return random_names

def generate_lastlatter():
    n_samples = 111
    names = generate_name(n_samples)
    idx = 1
    results = []
    for name in names:
        id = f"Lastletter_Q{idx}"
        question = f'Take the last letters of the words in "{name[0]} {name[1]}" and concatenate them.'
        context = []
        for x in name[0][:-1]:
            if x == name[0][-1]:
                continue
            sent = f'The last letter of "{name[0]}" is {x}.'
            context.append(sent)
        for x in name[1][:-1]:
            if x == name[1][-1]:
                continue
            sent = f'The last letter of "{name[1]}" is {x}.'
            context.append(sent)
        cor_sents = [f'The last letter of "{name[0]}" is {name[0][-1]}.', f'The last letter of "{name[1]}" is {name[1][-1]}.']
        context.extend(cor_sents)
        random.shuffle(context)
        context = " ".join(context)
        answer = f"{name[0][-1]}{name[1][-1]}"
        reason = " ".join(cor_sents) + " " + f'Concatenating them is {answer}.'
        msg = {'id':id, 'question':question, 'context':context, 'reason':reason, 'answer':answer}
        idx += 1
        results.append(msg)
    with jsonlines.open('/netdisk/ljc/code/faithful_cot/data/lastletter/dev.jsonl', mode='w') as writer:
        for item in results:
            writer.write(item)



def generate_coinflip():
    names = ['John', 'Emma', 'Michael', 'Olivia', 'William', 'Ava', 'James', 'Sophia', 'Benjamin', 'Isabella', 'Daniel', 'Mia', 'Joseph', 'Charlotte', 'David', 'Amelia', 'Alexander', 'Harper', 'Matthew', 'Evelyn', 'Andrew', 'Abigail', 'Christopher', 'Emily', 'Joshua', 'Elizabeth', 'Samantha', 'Nicholas', 'Madison', 'Jackson', 'Avery', 'Ryan', 'Chloe', 'Samuel', 'Ella', 'Jonathan', 'Grace', 'Nathan', 'Victoria', 'Christian', 'Scarlett', 'Tyler', 'Hannah', 'Dylan', 'Aria', 'Caleb', 'Lily', 'Mason', 'Zoe', 'Ethan', 'Penelope', 'Henry', 'Layla', 'Nora', 'Sebastian', 'Riley', 'Jack', 'Leah', 'Owen', 'Audrey', 'Wyatt', 'Stella', 'Luke', 'Ellie', 'Julian', 'Paisley', 'Levi', 'Skylar', 'Gabriel', 'Violet', 'Isaac', 'Claire', 'Lincoln', 'Brooklyn', 'Landon', 'Savannah', 'Carter', 'Genesis', 'Jayden', 'Bella', 'Naomi', 'Elena', 'Alexa', 'Serenity', 'Ariana', 'Maya', 'Valentina', 'Alice', 'Gabriella', 'Julia', 'Taylor', 'Aubrey', 'Autumn', 'Faith', 'Hazel', 'Hailey', 'Genesis', 'Piper', 'Willow', 'Eva', 'Quinn', 'Sadie', 'Nevaeh', 'Brielle', 'Peyton', 'Ruby', 'Sophie', 'Natalie', 'Serena', 'Luna', 'Vivian', 'Eleanor', 'Gianna', 'Isla', 'Clara', 'Lydia', 'Delilah', 'Alexandra', 'Elise', 'Hadley', 'Mackenzie', 'Kaylee', 'Sara', 'Jasmine', 'Malia', 'Adeline', 'Eliana', 'Charlie', 'Nadia', 'Juliette', 'Maria']
    n_samples = 303
    results = []
    for i in range(n_samples):
        id = f"CoinFlip_Q{i+1}"
        name = random.sample(names, 10)
        k = random.randint(1, 5)
        question = 'Is the coin still heads up?'
        context = []
        reason = ""
        for j in range(10):
            if j + 1 <= k:
                reason += f"The coin was flipped by {name[j]}. "
                context.append(f"{name[j]} flips the coin.")
            else:
                context.append(f"{name[j]} does not flip the coin.")
        if k % 2 == 1:
            answer = 'no'
            reason += f"So the coin was flipped {k} times, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up."
        else:
            answer = 'yes'
            reason += f"So the coin was flipped {k} times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up."
        random.shuffle(context)
        context = "A coin is heads up. " + " ".join(context)
        msg = {'id':id, 'context':context, 'question':question, 'answer':answer, 'reason':reason}
        results.append(msg)
        
    with jsonlines.open('/netdisk/ljc/code/faithful_cot/data/coinflip/dev.jsonl', mode='w') as writer:
        for item in results:
            writer.write(item)




generate_coinflip()