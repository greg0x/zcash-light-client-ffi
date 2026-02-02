//! FFI bindings for txid PIR action decryption.
//!
//! This module provides the FFI function to decrypt and store Orchard actions
//! reconstructed from PIR + compact block data.

use std::ffi::OsStr;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::slice;

use anyhow::anyhow;
use ffi_helpers::panic::catch_panic;
use orchard::note::ExtractedNoteCommitment;
use orchard::note::Nullifier as OrchardNullifier;
use orchard::note_encryption::{CompactAction, OrchardDomain};
use rand::rngs::OsRng;
use rusqlite::named_params;
use tracing::debug;
use zcash_client_backend::data_api::WalletRead;
use zcash_client_backend::TransferType;
use zcash_client_sqlite::{util::SystemClock, WalletDb};
use zcash_note_encryption::{
    try_note_decryption, try_output_recovery_with_ovk, EphemeralKeyBytes, ShieldedOutput,
    ENC_CIPHERTEXT_SIZE,
};
use zcash_protocol::consensus::Network;
use zcash_protocol::memo::MemoBytes;
use zip32::Scope;

/// Size of a complete reconstructed Orchard action for trial decryption.
/// Layout: nullifier(32) + cmx(32) + epk(32) + enc_ciphertext(580) + out_ciphertext(80) + cv(32) = 788
pub const PIR_ACTION_SIZE: usize = 788;

/// Offsets within the 788-byte action
const OFFSET_NULLIFIER: usize = 0;
const OFFSET_CMX: usize = 32;
const OFFSET_EPK: usize = 64;
const OFFSET_ENC_CIPHERTEXT: usize = 96;
const OFFSET_OUT_CIPHERTEXT: usize = 676;
const OFFSET_CV: usize = 756;

/// Reconstructed Orchard action from PIR + compact data.
struct ReconstructedAction {
    nullifier: OrchardNullifier,
    cmx: ExtractedNoteCommitment,
    ephemeral_key: EphemeralKeyBytes,
    enc_ciphertext: [u8; ENC_CIPHERTEXT_SIZE], // 580 bytes
    out_ciphertext: [u8; 80],
    cv: orchard::value::ValueCommitment,
}

impl ShieldedOutput<OrchardDomain, ENC_CIPHERTEXT_SIZE> for ReconstructedAction {
    fn ephemeral_key(&self) -> EphemeralKeyBytes {
        EphemeralKeyBytes(self.ephemeral_key.0)
    }

    fn cmstar_bytes(&self) -> [u8; 32] {
        self.cmx.to_bytes()
    }

    fn enc_ciphertext(&self) -> &[u8; ENC_CIPHERTEXT_SIZE] {
        &self.enc_ciphertext
    }
}

/// Decrypt and store Orchard actions from PIR data.
///
/// Takes an array of pre-merged 788-byte actions (merged from compact block + PIR data
/// by the caller) and performs trial decryption with wallet viewing keys.
///
/// # Action Layout (788 bytes each)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 32 | nullifier (from compact block) |
/// | 32 | 32 | cmx (from compact block) |
/// | 64 | 32 | ephemeral_key (from compact block) |
/// | 96 | 580 | enc_ciphertext (52 from compact + 528 from PIR) |
/// | 676 | 80 | out_ciphertext (from PIR) |
/// | 756 | 32 | cv - value commitment (from PIR) |
///
/// # Returns
///
/// Number of notes successfully decrypted and stored (>= 0), or -1 on error.
///
/// # Safety
///
/// - `db_data` must be valid for `db_data_len` bytes
/// - `txid` must point to 32 bytes
/// - `actions` must point to `action_count * 788` bytes
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zcashlc_decrypt_and_store_pir_actions(
    db_data: *const u8,
    db_data_len: usize,
    network_id: u32,
    txid: *const u8,
    _mined_height: u32,
    action_count: usize,
    actions: *const u8,
) -> i32 {
    let res = catch_panic(|| {
        // Validate inputs
        if db_data.is_null() || txid.is_null() {
            return Err(anyhow!("Required pointers are null"));
        }
        if action_count > 0 && actions.is_null() {
            return Err(anyhow!("Actions pointer is null"));
        }

        // Parse network
        let network = match network_id {
            0 => Network::TestNetwork,
            1 => Network::MainNetwork,
            _ => return Err(anyhow!("Invalid network_id: {}", network_id)),
        };

        // Open wallet database path
        let db_path = Path::new(OsStr::from_bytes(unsafe {
            slice::from_raw_parts(db_data, db_data_len)
        }));

        // Open WalletDb to get UFVKs (uses the proper librustzcash API)
        let db = WalletDb::for_path(db_path, network, SystemClock, OsRng)
            .map_err(|e| anyhow!("Failed to open wallet database: {}", e))?;

        // Get unified full viewing keys for trial decryption
        let ufvks = db
            .get_unified_full_viewing_keys()
            .map_err(|e| anyhow!("Failed to get viewing keys: {}", e))?;

        // Open raw sqlite connection for direct DB access
        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| anyhow!("Failed to open database: {}", e))?;

        // Parse txid
        let txid_bytes: [u8; 32] = unsafe { slice::from_raw_parts(txid, 32) }
            .try_into()
            .map_err(|_| anyhow!("Invalid txid length"))?;

        // Get the internal tx reference (id_tx) from the database
        let tx_ref: i64 = conn
            .query_row(
                "SELECT id_tx FROM transactions WHERE txid = ?",
                [&txid_bytes[..]],
                |row: &rusqlite::Row| row.get(0),
            )
            .map_err(|e| anyhow!("Transaction not found in database: {}", e))?;

        if ufvks.is_empty() {
            debug!("No viewing keys available for trial decryption");
            return Ok(0);
        }

        let actions_slice =
            unsafe { slice::from_raw_parts(actions, action_count * PIR_ACTION_SIZE) };

        let mut decrypted_count = 0i32;

        // Process each action
        for action_idx in 0..action_count {
            let action_data = &actions_slice[action_idx * PIR_ACTION_SIZE..][..PIR_ACTION_SIZE];

            // Parse the 788-byte action
            let action = match parse_action(action_data) {
                Ok(a) => a,
                Err(e) => {
                    debug!("Failed to parse action {}: {}", action_idx, e);
                    continue;
                }
            };

            // Create OrchardDomain from the compact action (uses nullifier to derive rho)
            // Build compact_action from raw bytes to avoid moving from action
            let enc_head: [u8; 52] = action_data[OFFSET_ENC_CIPHERTEXT..][..52]
                .try_into()
                .unwrap();
            let epk_bytes: [u8; 32] = action_data[OFFSET_EPK..][..32].try_into().unwrap();
            let compact_action = CompactAction::from_parts(
                action.nullifier, // Copy type
                action.cmx,       // Copy type
                EphemeralKeyBytes(epk_bytes),
                enc_head,
            );
            let domain = OrchardDomain::for_compact_action(&compact_action);

            // Try trial decryption with all UFVKs
            for (account_uuid, ufvk) in &ufvks {
                if let Some(orchard_fvk) = ufvk.orchard() {
                    // Try external IVK (incoming)
                    let ivk_external = orchard::keys::PreparedIncomingViewingKey::new(
                        &orchard_fvk.to_ivk(Scope::External),
                    );
                    if let Some((note, _, memo_bytes)) =
                        try_note_decryption(&domain, &ivk_external, &action)
                    {
                        let memo = MemoBytes::from_bytes(&memo_bytes)
                            .map_err(|_| anyhow!("Invalid memo bytes"))?;
                        debug!(
                            "Decrypted incoming note at action {} for account {:?}",
                            action_idx, account_uuid
                        );

                        store_orchard_note(
                            &conn,
                            tx_ref,
                            action_idx,
                            *account_uuid,
                            &note,
                            &memo,
                            TransferType::Incoming,
                        )?;
                        decrypted_count += 1;
                        break;
                    }

                    // Try internal IVK (change)
                    let ivk_internal = orchard::keys::PreparedIncomingViewingKey::new(
                        &orchard_fvk.to_ivk(Scope::Internal),
                    );
                    if let Some((note, _, memo_bytes)) =
                        try_note_decryption(&domain, &ivk_internal, &action)
                    {
                        let memo = MemoBytes::from_bytes(&memo_bytes)
                            .map_err(|_| anyhow!("Invalid memo bytes"))?;
                        debug!(
                            "Decrypted change note at action {} for account {:?}",
                            action_idx, account_uuid
                        );

                        store_orchard_note(
                            &conn,
                            tx_ref,
                            action_idx,
                            *account_uuid,
                            &note,
                            &memo,
                            TransferType::WalletInternal,
                        )?;
                        decrypted_count += 1;
                        break;
                    }

                    // Try OVK recovery (outgoing)
                    let ovk = orchard_fvk.to_ovk(Scope::External);
                    if let Some((note, _, memo_bytes)) = try_output_recovery_with_ovk(
                        &domain,
                        &ovk,
                        &action,
                        &action.cv,
                        &action.out_ciphertext,
                    ) {
                        let memo = MemoBytes::from_bytes(&memo_bytes)
                            .map_err(|_| anyhow!("Invalid memo bytes"))?;
                        debug!(
                            "Recovered outgoing note at action {} for account {:?}",
                            action_idx, account_uuid
                        );

                        store_orchard_note(
                            &conn,
                            tx_ref,
                            action_idx,
                            *account_uuid,
                            &note,
                            &memo,
                            TransferType::Outgoing,
                        )?;
                        decrypted_count += 1;
                        break;
                    }
                }
            }
        }

        debug!(
            "PIR: Decrypted {} notes from {} actions",
            decrypted_count, action_count
        );
        Ok(decrypted_count)
    });

    match res {
        Ok(count) => count,
        Err(_) => -1,
    }
}

/// Parse a 788-byte action into its components.
fn parse_action(data: &[u8]) -> anyhow::Result<ReconstructedAction> {
    if data.len() != PIR_ACTION_SIZE {
        return Err(anyhow!(
            "Invalid action size: {} (expected {})",
            data.len(),
            PIR_ACTION_SIZE
        ));
    }

    // Parse nullifier
    let nf_bytes: [u8; 32] = data[OFFSET_NULLIFIER..][..32].try_into().unwrap();
    let nullifier = OrchardNullifier::from_bytes(&nf_bytes);
    if nullifier.is_none().into() {
        return Err(anyhow!("Invalid nullifier"));
    }
    let nullifier = nullifier.unwrap();

    // Parse cmx
    let cmx_bytes: [u8; 32] = data[OFFSET_CMX..][..32].try_into().unwrap();
    let cmx = ExtractedNoteCommitment::from_bytes(&cmx_bytes);
    if cmx.is_none().into() {
        return Err(anyhow!("Invalid cmx"));
    }
    let cmx = cmx.unwrap();

    // Parse ephemeral key
    let epk_bytes: [u8; 32] = data[OFFSET_EPK..][..32].try_into().unwrap();

    // Parse enc_ciphertext (580 bytes)
    let enc_ciphertext: [u8; ENC_CIPHERTEXT_SIZE] = data[OFFSET_ENC_CIPHERTEXT..][..580]
        .try_into()
        .unwrap();

    // Parse out_ciphertext (80 bytes)
    let out_ciphertext: [u8; 80] = data[OFFSET_OUT_CIPHERTEXT..][..80].try_into().unwrap();

    // Parse cv (value commitment)
    let cv_bytes: [u8; 32] = data[OFFSET_CV..][..32].try_into().unwrap();
    let cv = orchard::value::ValueCommitment::from_bytes(&cv_bytes);
    if cv.is_none().into() {
        return Err(anyhow!("Invalid cv"));
    }
    let cv = cv.unwrap();

    Ok(ReconstructedAction {
        nullifier,
        cmx,
        ephemeral_key: EphemeralKeyBytes(epk_bytes),
        enc_ciphertext,
        out_ciphertext,
        cv,
    })
}

/// Store a decrypted Orchard note in the wallet database.
fn store_orchard_note(
    conn: &rusqlite::Connection,
    tx_ref: i64,
    action_index: usize,
    account_uuid: zcash_client_sqlite::AccountUuid,
    note: &orchard::note::Note,
    memo: &MemoBytes,
    transfer_type: TransferType,
) -> anyhow::Result<()> {
    // Get account_id from account UUID
    let account_id: i64 = conn
        .query_row(
            "SELECT id FROM accounts WHERE uuid = ?",
            [account_uuid.expose_uuid().as_bytes().as_slice()],
            |row: &rusqlite::Row| row.get(0),
        )
        .map_err(|e| anyhow!("Failed to find account: {}", e))?;

    let recipient = note.recipient();
    let diversifier = recipient.diversifier();
    let rseed = note.rseed();
    let is_change = matches!(transfer_type, TransferType::WalletInternal);

    // Encode key scope (0 = External, 1 = Internal)
    let key_scope: i64 = match transfer_type {
        TransferType::Incoming | TransferType::Outgoing => 0,
        TransferType::WalletInternal => 1,
    };

    // Convert memo to storage format (NULL if all zeros)
    let memo_bytes = memo.as_slice();
    let memo_value: Option<&[u8]> = if memo_bytes.iter().all(|&b| b == 0) {
        None
    } else {
        Some(memo_bytes)
    };

    conn.execute(
        "INSERT INTO orchard_received_notes (
            tx, action_index, account_id,
            diversifier, value, rho, rseed, memo,
            is_change, recipient_key_scope
        )
        VALUES (
            :tx, :action_index, :account_id,
            :diversifier, :value, :rho, :rseed, :memo,
            :is_change, :recipient_key_scope
        )
        ON CONFLICT (tx, action_index) DO UPDATE
        SET account_id = :account_id,
            diversifier = :diversifier,
            value = :value,
            rho = :rho,
            rseed = :rseed,
            memo = IFNULL(:memo, memo),
            is_change = MAX(:is_change, is_change),
            recipient_key_scope = :recipient_key_scope",
        named_params![
            ":tx": tx_ref,
            ":action_index": action_index as i64,
            ":account_id": account_id,
            ":diversifier": diversifier.as_array(),
            ":value": note.value().inner() as i64,
            ":rho": note.rho().to_bytes(),
            ":rseed": rseed.as_bytes(),
            ":memo": memo_value,
            ":is_change": is_change,
            ":recipient_key_scope": key_scope,
        ],
    )
    .map_err(|e| anyhow!("Failed to store orchard note: {}", e))?;

    debug!(
        "Stored {} note at action {} - value: {} zats",
        match transfer_type {
            TransferType::Incoming => "incoming",
            TransferType::WalletInternal => "change",
            TransferType::Outgoing => "outgoing",
        },
        action_index,
        note.value().inner()
    );

    Ok(())
}
