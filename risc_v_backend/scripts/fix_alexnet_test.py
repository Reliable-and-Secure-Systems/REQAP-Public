with open('risc_v_backend/verify_output.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = '            ("conv4.0",     8,     "8bit-d1",   1),   # K=3456, baseline'
new = '            ("conv3.0",     8,     "8bit-d1",   1),   # K=2304, baseline, uniform 8b'

if old in content:
    content = content.replace(old, new)
    with open('risc_v_backend/verify_output.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('OK: conv4.0 replaced with conv3.0')
else:
    print('ERROR: string not found')
    # Show lines 50-57
    for i, line in enumerate(content.splitlines()[49:57], start=50):
        print(f'{i}: {repr(line)}')
