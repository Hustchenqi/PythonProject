def line(file):
    for line in file:yield line
    yield '\n'

def blocks(file):
    block = []
    for line in line(file):
        if line.strip():
            block.append(line)
        elif block:
            yield ''.join(block).strip()
            block = []
            
