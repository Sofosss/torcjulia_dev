module logger

using Logging, Dates

const ORANGE    = "\x1b[38;5;214m"
const BOLD_GRAY = "\x1b[1;90m"
const CYAN      = "\x1b[36m"
const GREEN     = "\x1b[32m"
const YELLOW    = "\x1b[33m"
const BLUE      = "\x1b[34m"
const RED       = "\x1b[31m"
const RESET     = "\x1b[0m"

color_message(level, msg) = level == Logging.Debug ? GREEN*msg*RESET  :
                            level == Logging.Info  ? BLUE*msg*RESET   :
                            level == Logging.Warn  ? YELLOW*msg*RESET :
                            level == Logging.Error ? RED*msg*RESET    : msg

level_string(level) = level == Logging.Debug ? "DEBUG" :
                      level == Logging.Info  ? "INFO"  :
                      level == Logging.Warn  ? "WARNING" :
                      level == Logging.Error ? "ERROR" : string(level)

mutable struct TorcLogger <: AbstractLogger
    name::String
    min_level::LogLevel
end

Logging.min_enabled_level(l::TorcLogger) = l.min_level
Logging.shouldlog(l::TorcLogger, level, _m, g, id) = level >= l.min_level

function Logging.handle_message(l::TorcLogger, level, msg, _m, g, id, file, line; kwargs...)
    t = Dates.format(now(), "yyyy-mm-dd HH:MM:SS.sss")
    h = string(ORANGE*"["*ORANGE*l.name*" "*BOLD_GRAY*level_string(level)*BOLD_GRAY*"]"*RESET)
    println(CYAN*t*RESET, " ", h, " ", color_message(level, msg))
end

function init_logger(min_level::LogLevel)
    name = join(split(string(@__MODULE__), ".")[2:3], ".") |> lowercase
    global_logger(TorcLogger(name, min_level))
end

set_logger_level(level::LogLevel) = (current_logger().min_level = level)

end
