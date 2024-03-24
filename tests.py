import pymongo # pip3 install pymongo


# função para usar depois
def db_append_object_status(max_porcentagem_falhas, min_porcentagem_falhas, avg_porcentagem_falhas=""):
    myclient = pymongo.MongoClient("mongodb://mongouser:mongouser@vps51980.publiccloud.com.br:27017/")
    db = myclient["db"]
    colletion_rodape = db["objeto_rodape"]
    rodape_specs = { "nome": "rodape1",
            "max_porcentagem_falhas": max_porcentagem_falhas,
            "min_porcentagem_falhas": min_porcentagem_falhas,
            "med_porcentagem_falhas": avg_porcentagem_falhas,
            "caminho_frame_processado": "images/qualquer.png" }
    x = colletion_rodape.insert_one(rodape_specs)
    return x.inserted_id



myclient = pymongo.MongoClient("mongodb://mongouser:mongouser@vps51980.publiccloud.com.br:27017/") # Connection string para mongo direto na VPS

db = myclient["db"] # Utilizar banco padrao de uso
colletion_rodape = db["objeto_rodape"] # selecionar colletion

for x in colletion_rodape.find(): # lista todos os registros dentro da collection.
    print(x)