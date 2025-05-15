from Chat import Chat

chat = Chat()

while True:
    ask = input("Você: ")
    if ask.lower() in ['sair', 'exit', 'quit']:
        break
    resposta = chat.answer(ask)
    print("Bot:", resposta)
