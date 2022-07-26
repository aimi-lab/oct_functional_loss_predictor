SELECT "file"."filename" AS "FILENAME",
COALESCE("file"."remarks"->>'inputPath', "file"."filename") AS "FILEPATH",
"patient"."surname" AS "STUDYID",
"patient"."name" AS "USUBJID",
(CASE
	WHEN "dataset"."laterality" = 'L' THEN 'OS'
	WHEN "dataset"."laterality" = 'R' THEN 'OD'
ELSE '' END) AS "FOCID",
"dataset"."uuid" AS "OEXFN",
(CASE
	WHEN "dataset"."laterality" = 'L' THEN 'LEFT'
	WHEN "dataset"."laterality" = 'R' THEN 'RIGHT'
ELSE 'BILATERAL' END) AS "OELAT",
"dataset"."acquisitionDatetime" AS "OEDTC",
"dataset"."seriesDatetime" AS "SERIESDTC",
"study"."studyDatetime" AS "STUDYDTC",
"childLayer"."content" AS "CONTENT",
"childLayer"."metadata" AS "METADATA",
"childLayer"."spacing" AS "SPACING",
"childLayer"."scale" AS "SCALE",
"childLayer"."shape" AS "SHAPE"
FROM "patient"
INNER JOIN (
	SELECT "id", "uuid", "laterality", "patientId", "fromFileId", "device",
		"manufacturer", "acquisitionDatetime", "seriesDatetime", "studyId"
	FROM "dataset"
	INNER JOIN (
		SELECT "workbook_dataset"."datasetId" FROM "workbook"
		INNER JOIN "workbook_dataset" ON "workbook_dataset"."workbookId" = "workbook"."id"
		INNER JOIN "dataset" ON "dataset"."id" = "workbook_dataset"."datasetId"
		WHERE "workbook"."uuid" = 'd97ac189-584b-49af-9c82-2afe9487a61a'
	) "workbook" ON "workbook"."datasetId" = "dataset"."id"
) "dataset" ON "dataset"."patientId" = "patient"."id"
INNER JOIN "file" ON "file"."id" = "dataset"."fromFileId"
INNER JOIN "layer" ON "layer"."datasetId" = "dataset"."id"
INNER JOIN "dataset" "childDataset" ON "childDataset"."fromDatasetId" = "dataset"."id"
INNER JOIN "layer" "childLayer" ON "childLayer"."datasetId" = "childDataset"."id"
INNER JOIN "study" ON "study"."id" = "dataset"."studyId"
WHERE ("childLayer"."name" = 'projection')
GROUP BY "patient"."id",
"file"."id",
"dataset"."id",
"dataset"."uuid",
"dataset"."laterality",
"dataset"."acquisitionDatetime",
"dataset"."seriesDatetime",
"study"."uuid",
"study"."studyDatetime",
"dataset"."manufacturer",
"dataset"."device",
"childDataset"."id",
"childLayer"."id"
ORDER BY "dataset"."id", "childDataset"."id";