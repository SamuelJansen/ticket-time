ID = 'ID_TICKET'
TYPE = 'TIPOTICKET'
AREA_ID = 'ID_AREA'
GROUP_ID = 'ID_GRUPO_SERVICO'
UNITY_ID = 'ID_UNIDADE'
OPEN_TIME = 'MIN_ABERTO'
CREATED_AT = 'DT_CRIACAO'
REQUESTER_ID = 'IDQUEMABRIUTICKET'
SOLVER_ID = 'IDTECRESP'
OPEN_AT_THE_TIME = 'QTD_TOTAL_ABERTOS_GRUPO_SERV'
SOLVING_TIME = 'TEMPOMINUTO'

INPUT_COLUMNS = {
    AREA_ID: float
    ,
    GROUP_ID: float
    ,
    OPEN_TIME: float
    ,
    TYPE: float
    ,
    UNITY_ID: float
    # ,
    # ID: int
    ,
    CREATED_AT: str
    # ,
    # REQUESTER_ID: float
    # ,
    # SOLVER_ID: float
    ,
    OPEN_AT_THE_TIME: float
}
OUTPUT_COLUMN = {
    SOLVING_TIME: float
}

RELEVANT_COLUMNS = {
    **INPUT_COLUMNS,
    **OUTPUT_COLUMN
}

MAX = 'MAX'
MEAN = 'MEAN'

R2 = 'R2'
RMSE = 'RMSE'
