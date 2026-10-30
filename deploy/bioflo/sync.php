<?php
/**
 * CerebroSpheroid Model Synchronization Hub
 * Hosted on bioflo.in
 *
 * Strict Bounded Storage Policy:
 * Only stores `latest.pt` and `best.pt` (plus lightweight `manifest.json`).
 * Employs chunked uploads to bypass Cloudflare 100 MB payload limits and PHP upload caps.
 */

// Disable execution time limit for large stream transfers
set_time_limit(600);
ini_set('memory_limit', '256M');

$SECRET_TOKEN = 'CerebroSpheroid_Sync_Secure_2026';

// 1. Authorization Check
$auth_header = '';
if (isset($_SERVER['HTTP_AUTHORIZATION'])) {
    $auth_header = trim($_SERVER['HTTP_AUTHORIZATION']);
} elseif (isset($_SERVER['REDIRECT_HTTP_AUTHORIZATION'])) {
    $auth_header = trim($_SERVER['REDIRECT_HTTP_AUTHORIZATION']);
} elseif (function_exists('apache_request_headers')) {
    $requestHeaders = apache_request_headers();
    $requestHeaders = array_combine(array_map('ucwords', array_keys($requestHeaders)), array_values($requestHeaders));
    if (isset($requestHeaders['Authorization'])) {
        $auth_header = trim($requestHeaders['Authorization']);
    }
}

$token_provided = '';
if (preg_match('/Bearer\s+(.*)$/i', $auth_header, $matches)) {
    $token_provided = $matches[1];
} elseif (isset($_REQUEST['token'])) {
    $token_provided = $_REQUEST['token'];
}

if ($token_provided !== $SECRET_TOKEN) {
    http_response_code(401);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'Unauthorized: Invalid or missing sync token']);
    exit;
}

// 2. Storage Setup
$storage_dir = __DIR__ . '/data';
if (!is_dir($storage_dir)) {
    mkdir($storage_dir, 0755, true);
    // Protect data directory from direct unauthenticated browser downloads
    file_put_contents($storage_dir . '/.htaccess', "Order Deny,Allow\nDeny from all\n");
}

$action = isset($_REQUEST['action']) ? $_REQUEST['action'] : 'status';

// --------------------------------------------------------------------------------------------
// Action: STATUS
// --------------------------------------------------------------------------------------------
if ($action === 'status') {
    header('Content-Type: application/json');
    $manifest_file = $storage_dir . '/manifest.json';
    $manifest = file_exists($manifest_file) ? json_decode(file_get_contents($manifest_file), true) : [];

    $latest_exists = file_exists($storage_dir . '/latest.pt');
    $best_exists = file_exists($storage_dir . '/best.pt');

    echo json_encode([
        'status' => 'online',
        'has_latest' => $latest_exists,
        'has_best' => $best_exists,
        'latest_size' => $latest_exists ? filesize($storage_dir . '/latest.pt') : 0,
        'best_size' => $best_exists ? filesize($storage_dir . '/best.pt') : 0,
        'manifest' => $manifest,
        'server_time' => date('c'),
    ]);
    exit;
}

// --------------------------------------------------------------------------------------------
// Action: DOWNLOAD
// --------------------------------------------------------------------------------------------
if ($action === 'download') {
    $type = (isset($_GET['type']) && $_GET['type'] === 'best') ? 'best' : 'latest';
    $target_file = $storage_dir . '/' . $type . '.pt';

    if (!file_exists($target_file)) {
        http_response_code(404);
        header('Content-Type: application/json');
        echo json_encode(['error' => "Checkpoint {$type}.pt not found on server"]);
        exit;
    }

    $file_size = filesize($target_file);
    header('Content-Description: File Transfer');
    header('Content-Type: application/octet-stream');
    header('Content-Disposition: attachment; filename="cerebro_spheroid_v4_' . $type . '.pt"');
    header('Content-Length: ' . $file_size);
    header('Cache-Control: must-revalidate, no-cache, private');
    header('Pragma: no-cache');

    // Stream out in 2 MB chunks
    $handle = fopen($target_file, 'rb');
    if ($handle === false) {
        http_response_code(500);
        exit;
    }
    while (!feof($handle)) {
        echo fread($handle, 2 * 1024 * 1024);
        flush();
    }
    fclose($handle);
    exit;
}

// --------------------------------------------------------------------------------------------
// Action: UPLOAD_CHUNK (Cloudflare 100 MB Safe, Atomic Overwrite)
// --------------------------------------------------------------------------------------------
if ($action === 'upload_chunk') {
    header('Content-Type: application/json');

    $type = (isset($_POST['type']) && $_POST['type'] === 'best') ? 'best' : 'latest';
    $chunk_index = isset($_POST['chunk_index']) ? intval($_POST['chunk_index']) : 0;
    $total_chunks = isset($_POST['total_chunks']) ? intval($_POST['total_chunks']) : 1;
    $step = isset($_POST['step']) ? intval($_POST['step']) : 0;
    $loss = isset($_POST['loss']) ? floatval($_POST['loss']) : 0.0;
    $source = isset($_POST['source']) ? preg_replace('/[^a-zA-Z0-9_\-]/', '', $_POST['source']) : 'unknown';

    $part_file = $storage_dir . '/' . $type . '.pt.part';
    $final_file = $storage_dir . '/' . $type . '.pt';

    // Retrieve binary chunk payload
    $chunk_data = null;
    if (isset($_FILES['chunk']) && is_uploaded_file($_FILES['chunk']['tmp_name'])) {
        $chunk_data = file_get_contents($_FILES['chunk']['tmp_name']);
    } else {
        $chunk_data = file_get_contents('php://input');
    }

    if ($chunk_data === null || strlen($chunk_data) === 0) {
        http_response_code(400);
        echo json_encode(['error' => 'Empty chunk payload received']);
        exit;
    }

    // First chunk creates / truncates; subsequent chunks append
    $mode = ($chunk_index === 0) ? 'wb' : 'ab';
    $fp = fopen($part_file, $mode);
    if (!$fp) {
        http_response_code(500);
        echo json_encode(['error' => 'Unable to open file for writing']);
        exit;
    }
    fwrite($fp, $chunk_data);
    fclose($fp);

    // If final chunk, atomically replace the target checkpoint
    if ($chunk_index === $total_chunks - 1) {
        if (!rename($part_file, $final_file)) {
            http_response_code(500);
            echo json_encode(['error' => 'Failed to finalize atomic checkpoint write']);
            exit;
        }

        // Update manifest
        $manifest_file = $storage_dir . '/manifest.json';
        $manifest = file_exists($manifest_file) ? json_decode(file_get_contents($manifest_file), true) : [];
        $manifest[$type] = [
            'step' => $step,
            'loss' => $loss,
            'source' => $source,
            'updated_at' => date('c'),
            'size_bytes' => filesize($final_file),
            'md5' => md5_file($final_file)
        ];
        file_put_contents($manifest_file, json_encode($manifest, JSON_PRETTY_PRINT));

        echo json_encode([
            'success' => true,
            'status' => 'completed',
            'type' => $type,
            'chunk_index' => $chunk_index,
            'total_chunks' => $total_chunks,
            'manifest' => $manifest[$type]
        ]);
        exit;
    }

    echo json_encode([
        'success' => true,
        'status' => 'in_progress',
        'type' => $type,
        'chunk_index' => $chunk_index,
        'total_chunks' => $total_chunks
    ]);
    exit;
}

http_response_code(400);
header('Content-Type: application/json');
echo json_encode(['error' => "Unknown action '{$action}'"]);
