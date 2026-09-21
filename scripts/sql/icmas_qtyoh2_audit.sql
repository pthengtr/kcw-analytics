-- HQ (KSS) PARTS9 — ICMAS QTYOH2 change audit
-- Run once as SQL admin on KSS (WinRM Administrator + sqlcmd -E, or SSMS).
-- Logs every real QTYOH2 value change with SQL login / host / app name so we can
-- attribute stock-check SA, transfer, KACC, or unexplained writers.
--
-- Companion ODBC APP tags (set in kcw-api writers):
--   kcw-stock-check, kcw-transfer
-- Legacy KACC / other tools usually leave APP_NAME as the client default.

USE PARTS9;
GO

IF OBJECT_ID(N'dbo.ICMAS_QTYOH2_AUDIT', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ICMAS_QTYOH2_AUDIT (
        audit_id        BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        bcode           NVARCHAR(50)  NOT NULL,
        old_qtyoh2      FLOAT         NULL,
        new_qtyoh2      FLOAT         NULL,
        delta_qtyoh2    FLOAT         NULL,
        changed_at_utc  DATETIME2(3)  NOT NULL
            CONSTRAINT DF_ICMAS_QOH2_AUD_at DEFAULT (SYSUTCDATETIME()),
        login_name      NVARCHAR(128) NULL,
        original_login  NVARCHAR(128) NULL,
        host_name       NVARCHAR(128) NULL,
        app_name        NVARCHAR(128) NULL,
        program_name    NVARCHAR(128) NULL,
        client_interface NVARCHAR(64) NULL,
        session_id      INT           NULL,
        nest_level      SMALLINT      NULL,
        also_updated    NVARCHAR(200) NULL  -- related cols present in the same UPDATE SET
    );

    CREATE INDEX IX_ICMAS_QOH2_AUD_bcode_at
        ON dbo.ICMAS_QTYOH2_AUDIT (bcode, changed_at_utc DESC);

    CREATE INDEX IX_ICMAS_QOH2_AUD_at
        ON dbo.ICMAS_QTYOH2_AUDIT (changed_at_utc DESC);
END
GO

IF EXISTS (SELECT 1 FROM sys.database_principals WHERE name = N'python_reader')
    GRANT SELECT ON dbo.ICMAS_QTYOH2_AUDIT TO [python_reader];
IF EXISTS (SELECT 1 FROM sys.database_principals WHERE name = N'python_writer')
    GRANT SELECT ON dbo.ICMAS_QTYOH2_AUDIT TO [python_writer];
GO

CREATE OR ALTER TRIGGER dbo.trg_ICMAS_qtyoh2_audit
ON dbo.ICMAS
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    -- Only care about QTYOH2 in the SET list (skip master-only / location-only updates).
    IF NOT UPDATE(QTYOH2)
        RETURN;

    DECLARE @also NVARCHAR(200) =
        CONCAT(
            CASE WHEN UPDATE(QTYOH1) THEN N'QTYOH1,' ELSE N'' END,
            CASE WHEN UPDATE(QTYOHA) THEN N'QTYOHA,' ELSE N'' END,
            CASE WHEN UPDATE(QTYOHB) THEN N'QTYOHB,' ELSE N'' END,
            CASE WHEN UPDATE(QTYOHC) THEN N'QTYOHC,' ELSE N'' END,
            CASE WHEN UPDATE(LOCATION1) THEN N'LOCATION1,' ELSE N'' END,
            CASE WHEN UPDATE(LOCATION2) THEN N'LOCATION2,' ELSE N'' END,
            CASE WHEN UPDATE(DATEUPDATE) THEN N'DATEUPDATE,' ELSE N'' END,
            CASE WHEN UPDATE(DATELAST) THEN N'DATELAST,' ELSE N'' END,
            CASE WHEN UPDATE(DATEAUDIT) THEN N'DATEAUDIT,' ELSE N'' END,
            CASE WHEN UPDATE(COSTAVG) THEN N'COSTAVG,' ELSE N'' END,
            CASE WHEN UPDATE(COSTLAST) THEN N'COSTLAST,' ELSE N'' END
        );
    IF LEN(@also) > 0
        SET @also = LEFT(@also, LEN(@also) - 1);

    DECLARE @login NVARCHAR(128) = SUSER_SNAME();
    DECLARE @orig  NVARCHAR(128) = ORIGINAL_LOGIN();
    DECLARE @host  NVARCHAR(128) = HOST_NAME();
    DECLARE @app   NVARCHAR(128) = APP_NAME();
    DECLARE @spid  INT = @@SPID;
    DECLARE @nest  SMALLINT = TRIGGER_NESTLEVEL();
    DECLARE @prog  NVARCHAR(128) = NULL;
    DECLARE @iface NVARCHAR(64) = NULL;

    SELECT
        @prog = s.program_name,
        @iface = s.client_interface_name
    FROM sys.dm_exec_sessions AS s
    WHERE s.session_id = @spid;

    INSERT INTO dbo.ICMAS_QTYOH2_AUDIT (
        bcode, old_qtyoh2, new_qtyoh2, delta_qtyoh2,
        login_name, original_login, host_name, app_name,
        program_name, client_interface, session_id, nest_level, also_updated
    )
    SELECT
        LTRIM(RTRIM(CONVERT(NVARCHAR(50), i.BCODE))),
        CONVERT(FLOAT, d.QTYOH2),
        CONVERT(FLOAT, i.QTYOH2),
        CONVERT(FLOAT, i.QTYOH2) - CONVERT(FLOAT, d.QTYOH2),
        @login,
        @orig,
        @host,
        @app,
        @prog,
        @iface,
        @spid,
        @nest,
        @also
    FROM inserted AS i
    INNER JOIN deleted AS d
        ON LTRIM(RTRIM(CONVERT(NVARCHAR(50), i.BCODE)))
         = LTRIM(RTRIM(CONVERT(NVARCHAR(50), d.BCODE)))
    WHERE EXISTS (
        SELECT CONVERT(FLOAT, i.QTYOH2)
        EXCEPT
        SELECT CONVERT(FLOAT, d.QTYOH2)
    );
END
GO
