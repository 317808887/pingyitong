<#
.SYNOPSIS
    从 PPT 封面展示的软件图标生成应用图标与托盘图标资源。

.DESCRIPTION
    图标来源：`屏译通-基于本地大模型的桌面实时原位翻译系统.pptx` 封面页
    （slide1）中作为应用图标展示的 squircle 图形（1024×1024）。

    处理流程：
      1. 读取源图并把四角的白色背景镂空为透明（仅处理与边缘连通的白色区域，
         因此猫身上的白色毛发不受影响）；
      2. 输出 app_icon.png（1024×1024，带透明通道）作为设计源文件；
      3. 生成多尺寸 app_icon.ico（16/24/32/48/64/128/256），
         供任务栏、开始菜单、文件资源管理器使用；
      4. 生成三种状态的托盘图标（运行中 / 已暂停 / 异常）。

    本脚本是设计源文件的一部分，随源码交付，任何人都可以重新生成资源。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/make_icons.ps1 `
        -Source "path/to/icon.jpg" -OutputDir "assets/icons"
#>

[CmdletBinding()]
param(
    # 源图标文件（PPT 中提取的 squircle 图形，实际为 JPEG 编码）
    [Parameter(Mandatory = $true)]
    [string]$Source,

    # 输出目录
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    # 主图标边长
    [int]$MasterSize = 1024,

    # 四角镂空的颜色容差（0-255）。源图角落为纯白，容差设为 24 足够，
    # 过大会误伤猫身浅色毛发。
    [int]$FloodTolerance = 26
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

# ----------------------------------------------------------------------
# 内联 C#：高性能位图处理
#
# PowerShell 逐像素操作 1024×1024 需要数分钟，因此把关键算法放入 C#，
# 由 .NET JIT 执行，整体耗时降到毫秒级。
# ----------------------------------------------------------------------
if (-not ('IconToolkit' -as [type])) {
Add-Type -ReferencedAssemblies 'System.Drawing' -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;

public static class IconToolkit
{
    /// <summary>
    /// 把与图像四边连通的近似白色区域镂空为透明。
    ///
    /// 采用泛洪填充而不是全局白色键控：源图是一只白底三花猫，
    /// 全局键控会把猫身上的白色毛发一起抹掉；而从四角开始填充只会
    /// 影响与边缘连通的背景区域。
    /// </summary>
    public static Bitmap MakeCornersTransparent(Bitmap source, int tolerance)
    {
        int w = source.Width, h = source.Height;

        // 转为 32bppArgb，保证可写入 alpha 通道
        Bitmap result = new Bitmap(w, h, PixelFormat.Format32bppArgb);
        using (Graphics g = Graphics.FromImage(result))
        {
            g.DrawImage(source, new Rectangle(0, 0, w, h));
        }

        BitmapData data = result.LockBits(
            new Rectangle(0, 0, w, h), ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);

        try
        {
            int stride = data.Stride;
            byte[] buffer = new byte[stride * h];
            System.Runtime.InteropServices.Marshal.Copy(data.Scan0, buffer, 0, buffer.Length);

            bool[] visited = new bool[w * h];
            var queue = new Queue<int>();

            // 从四边所有"近似白色"的像素入队，覆盖四种角落形态
            for (int x = 0; x < w; x++)
            {
                TryEnqueue(buffer, stride, visited, queue, x, 0, w, tolerance);
                TryEnqueue(buffer, stride, visited, queue, x, h - 1, w, tolerance);
            }
            for (int y = 0; y < h; y++)
            {
                TryEnqueue(buffer, stride, visited, queue, 0, y, w, tolerance);
                TryEnqueue(buffer, stride, visited, queue, w - 1, y, w, tolerance);
            }

            while (queue.Count > 0)
            {
                int index = queue.Dequeue();
                int px = index % w;
                int py = index / w;

                // 镂空：alpha 置 0
                buffer[py * stride + px * 4 + 3] = 0;

                TryEnqueue(buffer, stride, visited, queue, px - 1, py, w, tolerance);
                TryEnqueue(buffer, stride, visited, queue, px + 1, py, w, tolerance);
                TryEnqueue(buffer, stride, visited, queue, px, py - 1, w, tolerance);
                TryEnqueue(buffer, stride, visited, queue, px, py + 1, w, tolerance);
            }

            System.Runtime.InteropServices.Marshal.Copy(buffer, 0, data.Scan0, buffer.Length);
        }
        finally
        {
            result.UnlockBits(data);
        }

        return result;
    }

    private static void TryEnqueue(
        byte[] buffer, int stride, bool[] visited, Queue<int> queue,
        int x, int y, int width, int tolerance)
    {
        if (x < 0 || y < 0 || x >= width) return;
        int height = visited.Length / width;
        if (y >= height) return;

        int index = y * width + x;
        if (visited[index]) return;

        int offset = y * stride + x * 4;
        byte b = buffer[offset];
        byte g = buffer[offset + 1];
        byte r = buffer[offset + 2];

        // 与纯白的距离在容差内才视为背景
        if (r < 255 - tolerance || g < 255 - tolerance || b < 255 - tolerance) return;

        visited[index] = true;
        queue.Enqueue(index);
    }

    /// <summary>以高质量插值缩放位图。</summary>
    public static Bitmap Resize(Bitmap source, int size)
    {
        Bitmap result = new Bitmap(size, size, PixelFormat.Format32bppArgb);
        using (Graphics g = Graphics.FromImage(result))
        {
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            g.SmoothingMode = SmoothingMode.HighQuality;
            g.CompositingQuality = CompositingQuality.HighQuality;
            g.Clear(Color.Transparent);
            g.DrawImage(source, new Rectangle(0, 0, size, size));
        }
        return result;
    }

    /// <summary>
    /// 生成去色版本，用于"已暂停"状态的托盘图标。
    /// </summary>
    public static Bitmap Desaturate(Bitmap source, float saturation, float tintA, float tintR, float tintG, float tintB)
    {
        int w = source.Width, h = source.Height;
        Bitmap result = new Bitmap(w, h, PixelFormat.Format32bppArgb);

        // 亮度加权矩阵实现去色；末行用于按比例叠加色调
        ColorMatrix matrix = new ColorMatrix(new float[][]
        {
            new float[] { 0.33f + 0.67f * saturation, 0.33f - 0.33f * saturation, 0.33f - 0.33f * saturation, 0, 0 },
            new float[] { 0.33f - 0.33f * saturation, 0.33f + 0.67f * saturation, 0.33f - 0.33f * saturation, 0, 0 },
            new float[] { 0.33f - 0.33f * saturation, 0.33f - 0.33f * saturation, 0.33f + 0.67f * saturation, 0, 0 },
            new float[] { 0, 0, 0, 1, 0 },
            new float[] { tintR * tintA, tintG * tintA, tintB * tintA, 0, 1 }
        });

        using (ImageAttributes attributes = new ImageAttributes())
        {
            attributes.SetColorMatrix(matrix);
            using (Graphics g = Graphics.FromImage(result))
            {
                g.Clear(Color.Transparent);
                g.DrawImage(source,
                    new Rectangle(0, 0, w, h),
                    0, 0, w, h,
                    GraphicsUnit.Pixel, attributes);
            }
        }
        return result;
    }

    /// <summary>
    /// 写出多尺寸 ICO 文件。
    ///
    /// ICO 结构：6 字节文件头 + N×16 字节目录项 + 各尺寸图像数据。
    /// 尺寸 &gt;= 256 的条目宽高字段写 0（这是 ICO 格式的规定）。
    /// 图像数据使用 PNG 编码（Windows Vista 起支持，Windows 10/11 完全支持），
    /// 相比 BMP 编码体积更小且自带透明通道。
    /// </summary>
    public static void WriteIco(Bitmap[] bitmaps, int[] sizes, string path)
    {
        var payloads = new List<byte[]>();

        foreach (Bitmap bitmap in bitmaps)
        {
            using (MemoryStream stream = new MemoryStream())
            {
                bitmap.Save(stream, ImageFormat.Png);
                payloads.Add(stream.ToArray());
            }
        }

        using (FileStream file = new FileStream(path, FileMode.Create, FileAccess.Write))
        using (BinaryWriter writer = new BinaryWriter(file))
        {
            writer.Write((ushort)0);                 // reserved
            writer.Write((ushort)1);                 // type: 1 = icon
            writer.Write((ushort)bitmaps.Length);     // image count

            int offset = 6 + bitmaps.Length * 16;
            for (int i = 0; i < bitmaps.Length; i++)
            {
                int size = sizes[i];
                writer.Write((byte)(size >= 256 ? 0 : size));  // width
                writer.Write((byte)(size >= 256 ? 0 : size));  // height
                writer.Write((byte)0);               // palette count
                writer.Write((byte)0);               // reserved
                writer.Write((ushort)1);             // color planes
                writer.Write((ushort)32);            // bits per pixel
                writer.Write((uint)payloads[i].Length);
                writer.Write((uint)offset);
                offset += payloads[i].Length;
            }

            foreach (byte[] payload in payloads)
            {
                writer.Write(payload);
            }
        }
    }
}
'@
}

# ----------------------------------------------------------------------
# 生成
# ----------------------------------------------------------------------

if (-not (Test-Path -LiteralPath $Source)) {
    throw "找不到源图标文件：$Source"
}
if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$sourcePath = (Resolve-Path -LiteralPath $Source).Path
Write-Host "源图标：$sourcePath"

# 载入源图并复制到内存位图，避免文件句柄被占用
$original = [System.Drawing.Image]::FromFile($sourcePath)
try {
    $raw = New-Object System.Drawing.Bitmap $original.Width, $original.Height,
        ([System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($raw)
    try {
        $graphics.DrawImage($original, 0, 0, $original.Width, $original.Height)
    } finally {
        $graphics.Dispose()
    }
    Write-Host "源图尺寸：$($original.Width)x$($original.Height)"
} finally {
    $original.Dispose()
}

# 1) 镂空四角背景
$master = [IconToolkit]::MakeCornersTransparent($raw, $FloodTolerance)
$raw.Dispose()

# 2) 归一化到主尺寸（源图非 1024 时也能得到规范输出）
if ($master.Width -ne $MasterSize) {
    $scaled = [IconToolkit]::Resize($master, $MasterSize)
    $master.Dispose()
    $master = $scaled
}

$pngPath = Join-Path $OutputDir 'app_icon.png'
$master.Save($pngPath, [System.Drawing.Imaging.ImageFormat]::Png)
Write-Host "已生成 $pngPath ($MasterSize x $MasterSize, 含透明通道)"

# 3) 多尺寸 ICO
$icoSizes = @(16, 24, 32, 48, 64, 128, 256)
[System.Drawing.Bitmap[]]$icoBitmaps = @()
foreach ($size in $icoSizes) {
    if ($size -eq $MasterSize) {
        $icoBitmaps += $master
    } else {
        $icoBitmaps += [IconToolkit]::Resize($master, $size)
    }
}

$icoPath = Join-Path $OutputDir 'app_icon.ico'
[IconToolkit]::WriteIco($icoBitmaps, $icoSizes, $icoPath)
$icoInfo = Get-Item -LiteralPath $icoPath
Write-Host ("已生成 {0} （{1} 个尺寸，{2:N0} 字节）" -f $icoPath, $icoSizes.Count, $icoInfo.Length)

# 4) 托盘图标（32×32）
$traySize = 32
$trayBase = [IconToolkit]::Resize($master, $traySize)

$trayActive = Join-Path $OutputDir 'tray_active.png'
$trayBase.Save($trayActive, [System.Drawing.Imaging.ImageFormat]::Png)

# 已暂停：去色，视觉上"安静下来"
$paused = [IconToolkit]::Desaturate($trayBase, 0.0, 0.0, 0, 0, 0)
$trayPaused = Join-Path $OutputDir 'tray_paused.png'
$paused.Save($trayPaused, [System.Drawing.Imaging.ImageFormat]::Png)
$paused.Dispose()

# 异常：保留少量饱和度并叠琥珀色，与"暂停"明确区分
$idle = [IconToolkit]::Desaturate($trayBase, 0.35, 0.35, 0.88, 0.63, 0.24)
$trayIdle = Join-Path $OutputDir 'tray_idle.png'
$idle.Save($trayIdle, [System.Drawing.Imaging.ImageFormat]::Png)
$idle.Dispose()

$trayBase.Dispose()

foreach ($size in $icoSizes) { }
foreach ($bitmap in $icoBitmaps) { $bitmap.Dispose() }
$master.Dispose()

Write-Host "已生成 $trayActive / tray_paused.png / tray_idle.png（$traySize x $traySize）"
Write-Host "图标资源生成完成。"
