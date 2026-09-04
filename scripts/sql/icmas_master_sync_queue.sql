-- HQ (KSS) PARTS9 — ICMAS master sync queue + trigger
-- Run once as SQL admin on KSS. Enables near-real-time HQ→SYP product push
-- via hq-ubuntu poller (see kcw-docs/ops/icmas-master-sync.md).
--
-- Trigger enqueues BCODE on INSERT or when master catalog/price columns change.
-- Does NOT fire for QTY*/LOCATION*-only updates (transfer / stock-check safe).

USE PARTS9;
GO

IF OBJECT_ID(N'dbo.ICMAS_MASTER_SYNC_QUEUE', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ICMAS_MASTER_SYNC_QUEUE (
        queue_id     BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        bcode        NVARCHAR(50)  NOT NULL,
        event_type   CHAR(1)       NOT NULL,  -- I = insert, U = update
        queued_at    DATETIME2(0)  NOT NULL CONSTRAINT DF_ICMAS_MSQ_queued DEFAULT (SYSUTCDATETIME()),
        status       NVARCHAR(16)  NOT NULL CONSTRAINT DF_ICMAS_MSQ_status DEFAULT (N'pending'),
        processed_at DATETIME2(0)  NULL,
        error_msg    NVARCHAR(1000) NULL,
        CONSTRAINT CK_ICMAS_MSQ_event CHECK (event_type IN ('I', 'U')),
        CONSTRAINT CK_ICMAS_MSQ_status CHECK (status IN (N'pending', N'done', N'error'))
    );

    CREATE INDEX IX_ICMAS_MSQ_pending
        ON dbo.ICMAS_MASTER_SYNC_QUEUE (status, queue_id)
        INCLUDE (bcode, event_type);

    CREATE INDEX IX_ICMAS_MSQ_bcode_pending
        ON dbo.ICMAS_MASTER_SYNC_QUEUE (bcode, status);
END
GO

-- Poller login (POS_MSSQL_WRITER_USERNAME)
GRANT SELECT, INSERT, UPDATE ON dbo.ICMAS_MASTER_SYNC_QUEUE TO [python_writer];
-- Optional read-only for diagnostics
IF EXISTS (SELECT 1 FROM sys.database_principals WHERE name = N'python_reader')
    GRANT SELECT ON dbo.ICMAS_MASTER_SYNC_QUEUE TO [python_reader];
GO

CREATE OR ALTER TRIGGER dbo.trg_ICMAS_master_sync_queue
ON dbo.ICMAS
AFTER INSERT, UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    IF TRIGGER_NESTLEVEL() > 1
        RETURN;

    DECLARE @is_insert BIT =
        CASE WHEN NOT EXISTS (SELECT 1 FROM deleted) THEN 1 ELSE 0 END;

    -- Skip stock/bin-only UPDATEs (transfer / stock-check).
    -- UPDATE(col) is true for all columns on INSERT, so only gate on UPDATE path.
    IF @is_insert = 0
       AND NOT (
            UPDATE(JOURMODE) OR UPDATE(XCODE) OR UPDATE(MCODE) OR UPDATE(PCODE)
         OR UPDATE(ACODE) OR UPDATE(DESCR) OR UPDATE(MODEL) OR UPDATE(BRAND)
         OR UPDATE(OEM) OR UPDATE(VENDOR) OR UPDATE(MAIN) OR UPDATE(SUB)
         OR UPDATE(PART) OR UPDATE(UI1) OR UPDATE(UI2) OR UPDATE(UI3)
         OR UPDATE(UI4) OR UPDATE(MTP2) OR UPDATE(MTP3) OR UPDATE(MTP4)
         OR UPDATE(STATUS) OR UPDATE(SERIAL) OR UPDATE(MIX) OR UPDATE(EXMPT)
         OR UPDATE(ISVAT) OR UPDATE(CODE1) OR UPDATE(CODE2) OR UPDATE(CODE3)
         OR UPDATE(CODE4) OR UPDATE(SIZE1) OR UPDATE(SIZE2) OR UPDATE(SIZE3)
         OR UPDATE(PRICELIST) OR UPDATE(DATELIST)
         OR UPDATE(PRICE1) OR UPDATE(PRICE2) OR UPDATE(PRICE3) OR UPDATE(PRICE4) OR UPDATE(PRICE5)
         OR UPDATE(MARKUP1) OR UPDATE(MARKUP2) OR UPDATE(MARKUP3) OR UPDATE(MARKUP4) OR UPDATE(MARKUP5)
         OR UPDATE(PRICEM1) OR UPDATE(PRICEM2) OR UPDATE(PRICEM3) OR UPDATE(PRICEM4) OR UPDATE(PRICEM5)
         OR UPDATE(PBDATE) OR UPDATE(PEDATE) OR UPDATE(PPRICE1) OR UPDATE(PPRICE2) OR UPDATE(PMTP2)
         OR UPDATE(COSTSET1) OR UPDATE(COSTSET2) OR UPDATE(COSTSET3) OR UPDATE(COSTSET4)
         OR UPDATE(DISCNT) OR UPDATE(DISCNT1) OR UPDATE(DISCNT2) OR UPDATE(DISCNT3) OR UPDATE(DISCNT4)
         OR UPDATE(COSTNET) OR UPDATE(COSTBEG1) OR UPDATE(COSTBEG2)
         OR UPDATE(REMARKS) OR UPDATE(CANCELED)
       )
        RETURN;

    ;WITH candidates AS (
        SELECT
            LTRIM(RTRIM(i.BCODE)) AS bcode,
            CASE WHEN @is_insert = 1 THEN 'I' ELSE 'U' END AS event_type
        FROM inserted i
        LEFT JOIN deleted d ON d.ID = i.ID
        WHERE LTRIM(RTRIM(ISNULL(i.BCODE, N''))) <> N''
          AND (
                @is_insert = 1
             -- REMARKS is ntext: cannot read inserted/deleted values; rely on UPDATE()
             OR (@is_insert = 0 AND UPDATE(REMARKS))
             OR ISNULL(i.JOURMODE, N'') <> ISNULL(d.JOURMODE, N'')
             OR ISNULL(i.XCODE, N'') <> ISNULL(d.XCODE, N'')
             OR ISNULL(i.MCODE, N'') <> ISNULL(d.MCODE, N'')
             OR ISNULL(i.PCODE, N'') <> ISNULL(d.PCODE, N'')
             OR ISNULL(i.ACODE, N'') <> ISNULL(d.ACODE, N'')
             OR ISNULL(i.DESCR, N'') <> ISNULL(d.DESCR, N'')
             OR ISNULL(i.MODEL, N'') <> ISNULL(d.MODEL, N'')
             OR ISNULL(i.BRAND, N'') <> ISNULL(d.BRAND, N'')
             OR ISNULL(i.OEM, N'') <> ISNULL(d.OEM, N'')
             OR ISNULL(i.VENDOR, N'') <> ISNULL(d.VENDOR, N'')
             OR ISNULL(i.MAIN, 0) <> ISNULL(d.MAIN, 0)
             OR ISNULL(i.SUB, 0) <> ISNULL(d.SUB, 0)
             OR ISNULL(i.PART, 0) <> ISNULL(d.PART, 0)
             OR ISNULL(i.UI1, N'') <> ISNULL(d.UI1, N'')
             OR ISNULL(i.UI2, N'') <> ISNULL(d.UI2, N'')
             OR ISNULL(i.UI3, N'') <> ISNULL(d.UI3, N'')
             OR ISNULL(i.UI4, N'') <> ISNULL(d.UI4, N'')
             OR ISNULL(i.MTP2, 0) <> ISNULL(d.MTP2, 0)
             OR ISNULL(i.MTP3, 0) <> ISNULL(d.MTP3, 0)
             OR ISNULL(i.MTP4, 0) <> ISNULL(d.MTP4, 0)
             OR ISNULL(i.STATUS, 0) <> ISNULL(d.STATUS, 0)
             OR ISNULL(i.SERIAL, N'') <> ISNULL(d.SERIAL, N'')
             OR ISNULL(i.MIX, N'') <> ISNULL(d.MIX, N'')
             OR ISNULL(i.EXMPT, N'') <> ISNULL(d.EXMPT, N'')
             OR ISNULL(i.ISVAT, N'') <> ISNULL(d.ISVAT, N'')
             OR ISNULL(i.CODE1, N'') <> ISNULL(d.CODE1, N'')
             OR ISNULL(i.CODE2, N'') <> ISNULL(d.CODE2, N'')
             OR ISNULL(i.CODE3, N'') <> ISNULL(d.CODE3, N'')
             OR ISNULL(i.CODE4, N'') <> ISNULL(d.CODE4, N'')
             OR ISNULL(i.SIZE1, N'') <> ISNULL(d.SIZE1, N'')
             OR ISNULL(i.SIZE2, N'') <> ISNULL(d.SIZE2, N'')
             OR ISNULL(i.SIZE3, N'') <> ISNULL(d.SIZE3, N'')
             OR ISNULL(i.PRICELIST, 0) <> ISNULL(d.PRICELIST, 0)
             OR ISNULL(CONVERT(VARCHAR(19), i.DATELIST, 120), '')
                <> ISNULL(CONVERT(VARCHAR(19), d.DATELIST, 120), '')
             OR ISNULL(i.PRICE1, 0) <> ISNULL(d.PRICE1, 0)
             OR ISNULL(i.PRICE2, 0) <> ISNULL(d.PRICE2, 0)
             OR ISNULL(i.PRICE3, 0) <> ISNULL(d.PRICE3, 0)
             OR ISNULL(i.PRICE4, 0) <> ISNULL(d.PRICE4, 0)
             OR ISNULL(i.PRICE5, 0) <> ISNULL(d.PRICE5, 0)
             OR ISNULL(i.MARKUP1, 0) <> ISNULL(d.MARKUP1, 0)
             OR ISNULL(i.MARKUP2, 0) <> ISNULL(d.MARKUP2, 0)
             OR ISNULL(i.MARKUP3, 0) <> ISNULL(d.MARKUP3, 0)
             OR ISNULL(i.MARKUP4, 0) <> ISNULL(d.MARKUP4, 0)
             OR ISNULL(i.MARKUP5, 0) <> ISNULL(d.MARKUP5, 0)
             OR ISNULL(i.PRICEM1, 0) <> ISNULL(d.PRICEM1, 0)
             OR ISNULL(i.PRICEM2, 0) <> ISNULL(d.PRICEM2, 0)
             OR ISNULL(i.PRICEM3, 0) <> ISNULL(d.PRICEM3, 0)
             OR ISNULL(i.PRICEM4, 0) <> ISNULL(d.PRICEM4, 0)
             OR ISNULL(i.PRICEM5, 0) <> ISNULL(d.PRICEM5, 0)
             OR ISNULL(CONVERT(VARCHAR(19), i.PBDATE, 120), '')
                <> ISNULL(CONVERT(VARCHAR(19), d.PBDATE, 120), '')
             OR ISNULL(CONVERT(VARCHAR(19), i.PEDATE, 120), '')
                <> ISNULL(CONVERT(VARCHAR(19), d.PEDATE, 120), '')
             OR ISNULL(i.PPRICE1, 0) <> ISNULL(d.PPRICE1, 0)
             OR ISNULL(i.PPRICE2, 0) <> ISNULL(d.PPRICE2, 0)
             OR ISNULL(i.PMTP2, 0) <> ISNULL(d.PMTP2, 0)
             OR ISNULL(i.COSTSET1, 0) <> ISNULL(d.COSTSET1, 0)
             OR ISNULL(i.COSTSET2, 0) <> ISNULL(d.COSTSET2, 0)
             OR ISNULL(i.COSTSET3, 0) <> ISNULL(d.COSTSET3, 0)
             OR ISNULL(i.COSTSET4, 0) <> ISNULL(d.COSTSET4, 0)
             OR ISNULL(i.DISCNT, 0) <> ISNULL(d.DISCNT, 0)
             OR ISNULL(i.DISCNT1, 0) <> ISNULL(d.DISCNT1, 0)
             OR ISNULL(i.DISCNT2, 0) <> ISNULL(d.DISCNT2, 0)
             OR ISNULL(i.DISCNT3, 0) <> ISNULL(d.DISCNT3, 0)
             OR ISNULL(i.DISCNT4, 0) <> ISNULL(d.DISCNT4, 0)
             OR ISNULL(i.COSTNET, 0) <> ISNULL(d.COSTNET, 0)
             OR ISNULL(i.COSTBEG1, 0) <> ISNULL(d.COSTBEG1, 0)
             OR ISNULL(i.COSTBEG2, 0) <> ISNULL(d.COSTBEG2, 0)
             OR ISNULL(i.CANCELED, N'') <> ISNULL(d.CANCELED, N'')
          )
    )
    INSERT INTO dbo.ICMAS_MASTER_SYNC_QUEUE (bcode, event_type)
    SELECT c.bcode, c.event_type
    FROM candidates c
    WHERE NOT EXISTS (
        SELECT 1
        FROM dbo.ICMAS_MASTER_SYNC_QUEUE q
        WHERE q.bcode = c.bcode
          AND q.status = N'pending'
    );
END
GO
