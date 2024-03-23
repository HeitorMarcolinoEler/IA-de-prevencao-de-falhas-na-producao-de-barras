import pymongo # pip3 install pymongo

myclient = pymongo.MongoClient("mongodb://mongouser:mongouser@vps51980.publiccloud.com.br:27017/") # Connection string para mongo direto na VPS

db = myclient["db"] # Utilizar banco padrao de uso
colletion_rodape = db["objeto_rodape"] # selecionar colletion

'''
# dicionario com os campos necessarios, isso é só um exemplo inicial, pode adicionar ou remover.

rodape_specs = { "nome": "rodape1",
          "max_porcentagem_falhas": "30.50",
          "min_porcentagem_falhas": "11.90",
          "med_porcentagem_falhas": "20",
          "caminho_frame_processado": "images/qualquer.png" }

x = colletion_rodape.insert_one(rodape_specs)
print(x.inserted_id)
'''

for x in colletion_rodape.find(): # lista todos os registros dentro da collection.
    print(x)