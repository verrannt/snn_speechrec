def prints(content, status):
    if status is 0:
        msg = '.'
    elif status is 1:
        msg = 'DONE'
    elif status is 2:
        msg = 'WARN'
    elif status is 3:
        msg = 'FAIL'
    print(f'\n[{msg}]\t{content}\n')
