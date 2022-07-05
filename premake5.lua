require("vstudio")

-- CUDA 
premake.api.register {
    name = "cudaRelocatableCode",
    scope = "config",
    kind = "boolean"
}

premake.api.register {
    name = "cudaExtensibleWholeProgram",
    scope = "config",
    kind = "boolean"
}

premake.api.register {
    name = "cudaCompilerOptions",
    scope = "config",
    kind = "table"
}

premake.api.register {
    name = "cudaFastMath",
    scope = "config",
    kind = "boolean"
}

premake.api.register {
    name = "cudaVerbosePTXAS",
    scope = "config",
    kind = "boolean"
}

premake.api.register {
    name = "cudaMaxRegCount",
    scope = "config",
    kind = "string"
}

premake.api.register {
    name = "cudaFiles",
    scope = "config",
    kind = "table"
}

local function writeBoolean(property, value)
    if value == true or value == "On" then
        premake.w('\t<' .. property .. '>true</' .. property .. '>')
    elseif value == false or value == "Off" then
        premake.w('\t<' .. property .. '>false</' .. property .. '>')
    end
end

local function writeString(property, value)
    if value ~= nil and value ~= '' then
        premake.w('\t<' .. property .. '>' .. value .. '</' .. property .. '>')
    end
end

local function writeTableAsOneString(property, values)
    if values ~= nil then
        writeString(property, table.concat(values, ' '))
    end
end

local function addCompilerProps(cfg)
    premake.w('<CudaCompile>')

    -- Determine architecture to compile for
    if cfg.architecture == "x86_64" or cfg.architecture == "x64" then
        premake.w('\t<TargetMachinePlatform>64</TargetMachinePlatform>')
    elseif cfg.architecture == "x86" then
        premake.w('\t<TargetMachinePlatform>32</TargetMachinePlatform>')
    else
        error("Unsupported Architecture")
    end

    -- Set XML tags to their requested values 
    premake.w('<CodeGeneration></CodeGeneration>')
    writeBoolean('GenerateRelocatableDeviceCode', cfg.cudaRelocatableCode)
    writeBoolean('ExtensibleWholeProgramCompilation', cfg.cudaExtensibleWholeProgram)
    writeBoolean('FastMath', cfg.cudaFastMath)
    writeBoolean('PtxAsOptionV', cfg.cudaVerbosePTXAS)
    writeTableAsOneString('AdditionalOptions', cfg.cudaCompilerOptions)
    writeString('MaxRegCount', cfg.cudaMaxRegCount)

    premake.w('</CudaCompile>')
end

premake.override(premake.vstudio.vc2010.elements, "itemDefinitionGroup", function(oldfn, cfg)
    local items = oldfn(cfg)
    table.insert(items, addCompilerProps)
    return items
end)

local function inlineFileWrite(value)
    premake.w('\t<CudaCompile ' .. 'Include=' .. string.escapepattern('"') .. path.getabsolute(value) ..
                  string.escapepattern('"') .. '/>')
end

local function checkForGlob(value)
    matchingFiles = os.matchfiles(value)
    if matchingFiles ~= null then
        table.foreachi(matchingFiles, inlineFileWrite)
    end
end

local function addCUDAFiles(cfg)
    if cfg.cudaFiles ~= null then
        premake.w('<ItemGroup>')
        table.foreachi(cfg.cudaFiles, checkForGlob)
        premake.w('</ItemGroup>')
    end
end

premake.override(premake.vstudio.vc2010.elements, "project", function(oldfn, cfg)
    local items = oldfn(cfg)
    table.insert(items, addCUDAFiles)
    return items
end)

-- Workspace
workspace "FluidEngine"
    architecture "x64"

    configurations
    {
        "Debug",
        "Release"
    }

    startproject "FluidEngine"

outputdir = "{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

-- Include directories relative to root folder (solution directory).
IncludeDir = {}
IncludeDir["GLFW"] = "FluidEngine/Vendor/GLFW/include"
IncludeDir["Glad"] = "FluidEngine/Vendor/Glad/include"
IncludeDir["ImGui"]= "FluidEngine/Vendor/imgui"
IncludeDir["glm"]  = "FluidEngine/Vendor/glm"
IncludeDir["entt"] = "FluidEngine/Vendor/entt/include"
IncludeDir["cereal"] = "FluidEngine/Vendor/cereal"

include "FluidEngine/Vendor/GLFW"
include "FluidEngine/Vendor/Glad"
include "FluidEngine/Vendor/imgui"

project "FluidEngine"
    location "FluidEngine"
    kind "ConsoleApp"
    language "C++"

    targetdir "bin/%{cfg.buildcfg}"

    pchheader "pch.h"
    pchsource "FluidEngine/src/pch.cpp"

    buildcustomizations "BuildCustomizations/CUDA 11.7"
    cudaMaxRegCount "0"

    cudaCompilerOptions 
    {
        -- "-arch=sm_50"
         -- "-arch=sm_10"
    }   

    files
    {
        "%{prj.name}/res/**.*",
        "%{prj.name}/src/**.h",
        "%{prj.name}/src/**.cpp",
        "%{prj.name}/src/**.txt",
        "%{prj.name}/src/**.inl",
        "%{prj.name}/Vendor/glm/**.hpp",
        "%{prj.name}/Vendor/glm/**.inl",
        "%{prj.name}/Vendor/cereal/**.hpp",
    }

    cudaFiles 
    {
        "**.cu",
        "**.cuh"
    }

    includedirs
    {
        "%{prj.name}/$(ProjectDir)src",
        "%{IncludeDir.GLFW}",
        "%{IncludeDir.Glad}",
        "%{IncludeDir.ImGui}",
        "%{IncludeDir.glm}",
        "%{IncludeDir.entt}",
        "%{IncludeDir.cereal}"
    }

    links
    {
        "GLFW",
        "Glad",
        "ImGui",
        "opengl32.lib",
        "cudart.lib"
    }

    filter "system:windows"
        cppdialect "C++20"
        staticruntime "On"
        systemversion "latest"

        defines
        {
            "GLFW_INCLUDE_NONE"
        }

    filter "configurations:Debug"
        defines 
        {
            "DEBUG" 
        }

        symbols "On"
        staticruntime "off"
  
     filter "configurations:Release"
        defines 
        {
            "NDEBUG" 
        }
        
        optimize "Full"
        staticruntime "off"
        cudaFastMath "On"
