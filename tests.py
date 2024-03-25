import pymongo # pip3 install pymongo





myclient = pymongo.MongoClient("mongodb://mongouser:mongouser@vps51980.publiccloud.com.br:27017/") # Connection string para mongo direto na VPS

db = myclient["db"] # Utilizar banco padrao de uso
colletion_rodape = db["objeto_rodape"] # selecionar colletion

for x in colletion_rodape.find(): # lista todos os registros dentro da collection.
    print(x)